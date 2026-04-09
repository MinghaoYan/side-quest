import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.tensorboard_utils import _TensorboardAdapter
from slime.utils.wandb_utils import init_wandb_primary

_OOM_SENTINEL = object()


def _is_training_oom_error(exc) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    oom_markers = (
        "outofmemoryerror",
        "cuda out of memory",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
    )
    return any(marker in text for marker in oom_markers)


def _ray_get_train_or_quit(object_refs, *, role: str, rollout_id: int):
    try:
        return ray.get(object_refs)
    except Exception as exc:
        if not _is_training_oom_error(exc):
            raise
        print(
            (
                f"[TRAIN-OOM] role={role} rollout_id={rollout_id} "
                "Quitting job cleanly after CUDA OOM during training.\n"
                f"{exc}"
            ),
            flush=True,
        )
        try:
            ray.shutdown()
        except Exception:
            pass
        return _OOM_SENTINEL


def train(args):
    # allocate the GPUs
    pgs = create_placement_groups(args)
    wandb_run_id = init_wandb_primary(args)

    if args.use_tensorboard:
        _TensorboardAdapter(args)

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, wandb_run_id=wandb_run_id)

    # create the rollout manager, with sglang engines inside.
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"], wandb_run_id=wandb_run_id)

    actor_model.set_rollout_manager(rollout_manager)

    if args.offload:
        ray.get(rollout_manager.offload.remote())
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    if args.offload:
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # TODO extract the duplicated eval logic
        if args.eval_interval is not None and rollout_id == 0:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload:
            ray.get(rollout_manager.offload.remote())

        # Train step: internally handles debug_rollout_only mode
        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps:
                actor_results = _ray_get_train_or_quit(
                    actor_model.async_train(rollout_id, rollout_data_ref),
                    role="actor",
                    rollout_id=rollout_id,
                )
                if actor_results is _OOM_SENTINEL:
                    return
            critic_results = _ray_get_train_or_quit(
                critic_train_handle,
                role="critic",
                rollout_id=rollout_id,
            )
            if critic_results is _OOM_SENTINEL:
                return
        else:
            actor_results = _ray_get_train_or_quit(
                actor_model.async_train(rollout_id, rollout_data_ref),
                role="actor",
                rollout_id=rollout_id,
            )
            if actor_results is _OOM_SENTINEL:
                return


        if args.save_interval is not None and (
            (rollout_id + 1) % args.save_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            actor_model.save_model(rollout_id)
            if args.use_critic:
                critic_model.save_model(rollout_id)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))
            elif args.evolving_gym or getattr(args, "pacevolve_gym", False):
                ray.get(rollout_manager.save.remote(rollout_id))
            else :
                assert False, "None of args.rollout_global_dataset, args.evolving_gym, args.pacevolve_gym is set."

        if args.offload:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()

            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

        actor_model.update_weights()

        if args.offload:
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(rollout_manager.eval.remote(rollout_id))


if __name__ == "__main__":
    args = parse_args()
    train(args)
