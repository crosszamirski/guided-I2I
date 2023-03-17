import argparse
import os
import blobfile as bf
import torch as th
import torch.nn.functional as F
from torch.optim import AdamW

from guided_diffusion import logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict


def main():
    args = create_argparser().parse_args()
    print(args)
    logger.configure()
    logger.log("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )


    logger.log("creating data loader...")
    data = load_data(data_dir="/.../target2_SET_C_train_actives.csv",
                 batch_size = 8,
                     )

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(opt_checkpoint)

    logger.log("training classifier model...")

    def forward_backward_log(data_loader, prefix="train"):
        
        batch, extra = next(data_loader)
        batch_bf, batch_cp = batch.split(split_size=[3,5], dim=1)
        labels = extra["y"]
        if args.noised:
            t, _ = schedule_sampler.sample(batch_cp.shape[0])
            batch_cp = diffusion.q_sample(batch_cp, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long)
        
        batch = th.cat((batch_bf, batch_cp), dim=1)
        print("Microbatching")
        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            print(i)
            logits = model(sub_batch, gammas=sub_t)
            loss = F.cross_entropy(logits, sub_labels, reduction="none")
            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )
            log_loss_dict(diffusion, sub_t, losses)

            del losses
            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))
           
    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size,
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        print(step)
        forward_backward_log(data_loader = data, prefix="train")
        mp_trainer.optimize(opt)
        if not step % args.log_interval:
            logger.dumpkvs()
        if (
            step
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    th.save(
        mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
        os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
    )
    th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.05,
        anneal_lr=True,
        image_size=512,
        batch_size=8,
        microbatch=1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=10,
        save_interval=10,
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
