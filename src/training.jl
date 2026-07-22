# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Main training entry point for SynapticDistill.
"""

"""
    ModelStep

Callable interface for model forward steps used by `train_step!`.

A model step is any callable object with the signature
`model_step(model, spikes::SpikeBatch) -> output`. The returned `output` is
passed unchanged to the caller-provided loss function.
"""
abstract type ModelStep end

# A placeholder for a default optimizer.
function default_optimizer()
    # In a real scenario, this would return a configured optimizer from a library like Optimisers.jl.
    return (params, grads) -> params .-= 0.001f0 .* grads
end

function validate_model_step(model_step)
    model_step === nothing && throw(ArgumentError("`forward_fn` is required; pass a callable `(model, spikes::SpikeBatch) -> output` to `train_step!`."))
    !(model_step isa Function) && !hasmethod(model_step, Tuple{Any, SpikeBatch}) && throw(ArgumentError("`forward_fn` must be callable as `(model, spikes::SpikeBatch) -> output`."))
    return model_step
end

"""
    train_step!(model, spikes::SpikeBatch, loss_fn; forward_fn, rule=:eprop, kwargs...)
    train_step!(model, spikes::SpikeBatch, loss_fn, model_step; rule=:eprop, kwargs...)

Perform one online training step using the chosen rule.

- `model`: Your SNN model.
- `spikes`: A `SpikeBatch` containing spike trains and optional targets.
- `loss_fn`: A function that takes the model output and computes a scalar loss.
- `forward_fn` / `model_step`: A caller-provided callable with signature
  `(model, spikes::SpikeBatch) -> output`.

Returns: A tuple of `(updated_model, TrainingState)`.
"""
function train_step!(model, spikes::SpikeBatch, loss_fn;
                     forward_fn = nothing,
                     rule::Symbol = :eprop,
                     optimizer = default_optimizer(),
                     kwargs...)

    model_step = validate_model_step(forward_fn)

    # 1. Forward pass through the caller-provided model step.
    output = model_step(model, spikes)

    # 2. Compute the loss using the user-provided function.
    loss, grads = Zygote.withgradient(() -> loss_fn(output), Zygote.params(model))

    # 3. Apply the chosen learning rule.
    if rule == :eprop
        # The `update_eprop!` function will calculate eligibility traces and gradients.
        # update_eprop!(model, spikes, loss, output; kwargs...)
        println("Applying e-prop rule (not fully implemented).")
    elseif rule == :ottt
        # update_ottt!(model, spikes, loss; kwargs...)
        println("Applying OTTT rule (not fully implemented).")
    else
        error("Unknown training rule: `$rule`")
    end

    # 4. Apply gradients (this is a simplified view).
    # In a real implementation, the rule-specific function would return gradients
    # to be applied here.
    # optimizer(Zygote.params(model), grads)

    # 5. Return the updated model and training state.
    state = TrainingState(loss=loss, gradients=grads)
    return model, state
end

function train_step!(model, spikes::SpikeBatch, loss_fn, model_step;
                     rule::Symbol = :eprop,
                     optimizer = default_optimizer(),
                     kwargs...)
    return train_step!(model, spikes, loss_fn;
                       forward_fn = model_step,
                       rule = rule,
                       optimizer = optimizer,
                       kwargs...)
end
