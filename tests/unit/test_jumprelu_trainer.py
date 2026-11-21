import torch as t

from dictionary_learning.trainers.jumprelu import JumpReluTrainer


def test_l0_threshold_grad_includes_preacts_just_below_threshold() -> None:
    """The L0 pseudo-derivative w.r.t. the threshold uses the pre-activations
    (eq. 10 of Rajamanoharan et al. 2024), so a pre-activation just below the
    threshold must also push on it, not only the ones that already fired."""
    trainer = JumpReluTrainer(
        steps=10,
        warmup_steps=0,
        activation_dim=1,
        dict_size=1,
        layer=0,
        lm_name="test",
        bandwidth=0.001,
        sparsity_warmup_steps=None,
        target_l0=2.0,
    )
    with t.no_grad():
        trainer.ae.W_enc.fill_(1.0)
        trainer.ae.b_enc.zero_()
        # zero decoder: the reconstruction term gives the threshold no gradient,
        # so everything left comes from the sparsity term
        trainer.ae.W_dec.zero_()
        trainer.ae.b_dec.zero_()
        trainer.ae.threshold.fill_(0.05)

    x = t.tensor([[0.0499]])  # pre-activation just below the threshold
    trainer.loss(x, step=0).backward()

    # d(sparsity)/d(l0) = 2 * (0 / 2 - 1) / 2 = -1, and the rectangle kernel
    # gives d(l0)/d(threshold) = -1 / bandwidth, so the gradient is +1000
    assert t.allclose(trainer.ae.threshold.grad, t.tensor([1000.0]), rtol=1e-3)

    # far outside the kernel window, there is still no gradient
    trainer.ae.threshold.grad = None
    trainer.loss(t.tensor([[0.0]]), step=0).backward()
    assert t.allclose(trainer.ae.threshold.grad, t.tensor([0.0]))


def test_l0_does_not_count_inactive_features_for_negative_threshold() -> None:
    """With a negative threshold, an inactive feature has f == 0 > threshold,
    so counting on f would report it as firing."""
    trainer = JumpReluTrainer(
        steps=10,
        warmup_steps=0,
        activation_dim=1,
        dict_size=1,
        layer=0,
        lm_name="test",
        sparsity_penalty=1.0,
        sparsity_warmup_steps=None,
        target_l0=1.0,
    )
    with t.no_grad():
        trainer.ae.W_enc.fill_(1.0)
        trainer.ae.b_enc.zero_()
        trainer.ae.W_dec.zero_()
        trainer.ae.b_dec.zero_()
        trainer.ae.threshold.fill_(-0.01)

    x = t.tensor([[-0.5]])  # below the threshold: the feature does not fire
    log = trainer.loss(x, step=0, logging=True)
    # l0 == 0 against target 1 gives sparsity loss 1; recon loss is x**2
    assert abs(log.losses["loss"] - (0.25 + 1.0)) < 1e-5
