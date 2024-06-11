import torch
from torch import Tensor

from torch.optim.optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _view_as_real,
                        _default_to_fused_or_foreach, _get_scalar_dtype, _differentiable_doc,
                        _foreach_doc, _maximize_doc)
from typing import List, Optional

__all__ = ["DeltaAdaGrad"]

MAX_GPI_VALUE = 0.1
MIN_GPI_VALUE = 0.001


class DeltaAdaGrad(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        foreach: Optional[bool] = None,
        *,
        maximize: bool = False,
        differentiable: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= lr_decay:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                f"Invalid initial_accumulator_value value: {initial_accumulator_value}"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                init_value = (
                    complex(initial_accumulator_value, initial_accumulator_value)
                    if torch.is_complex(p)
                    else initial_accumulator_value
                )
                state["sum"] = torch.full_like(
                    p, init_value, memory_format=torch.preserve_format
                )

        self.prev_loss = None

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)

        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]), dtype=_get_scalar_dtype())

    # def share_memory(self):
    #     for group in self.param_groups:
    #         for p in group["params"]:
    #             state = self.state[p]
    #             state["sum"].share_memory_()

    def _init_group(self, group, params_with_grad, grads, state_sums, state_steps):
        has_sparse_grad, has_complex = False, False
        for p in group["params"]:
            if p.grad is not None:
                has_sparse_grad |= p.grad.is_sparse
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                state_sums.append(state["sum"])
                state_steps.append(state["step"])

        return has_sparse_grad, has_complex


    @_use_grad_for_differentiable
    def step(self, closure=None, loss=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if loss is None:
            raise RuntimeError("DeltaAdaGrad requires loss to be specified.")

        loss = loss.to("cpu").detach()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            state_sums = []
            state_steps = []

            has_sparse_grad, has_complex = self._init_group(group, params_with_grad, grads, state_sums, state_steps)

            deltaadagrad(
                params_with_grad,
                grads,
                state_sums,
                state_steps,
                loss,
                prev_loss=self.prev_loss,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                lr_decay=group["lr_decay"],
                eps=group["eps"],
                has_sparse_grad=has_sparse_grad,
                foreach=group["foreach"],
                maximize=group["maximize"],
                differentiable=group["differentiable"],
                has_complex=has_complex,
            )

        self.prev_loss = loss

        return loss


def deltaadagrad(
    params: List[Tensor],
    grads: List[Tensor],
    state_sums: List[Tensor],
    state_steps: List[Tensor],
    loss: Tensor,
    prev_loss: Tensor = None,
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting these as kwargs for now as functional API is compiled by torch/distributed/optim
    has_sparse_grad: bool = None,
    foreach: Optional[bool] = None,
    differentiable: bool = False,
    has_complex: bool = False,
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    maximize: bool,
):
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adagrad
    else:
        # func = _single_tensor_adagrad
        raise RuntimeError("DeltaAdaGrad does not support single tensor")

    func(
        params,
        grads,
        state_sums,
        state_steps,
        loss,
        prev_loss,
        lr=lr,
        weight_decay=weight_decay,
        lr_decay=lr_decay,
        eps=eps,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        differentiable=differentiable,
        has_complex=has_complex,
    )


# def _make_sparse(grad, grad_indices, values):
#     size = grad.size()
#     if grad_indices.numel() == 0 or values.numel() == 0:
#         return torch.empty_like(grad)
#     return torch.sparse_coo_tensor(grad_indices, values, size)


# def _single_tensor_adagrad(
#     params: List[Tensor],
#     grads: List[Tensor],
#     state_sums: List[Tensor],
#     state_steps: List[Tensor],
#     *,
#     lr: float,
#     weight_decay: float,
#     lr_decay: float,
#     eps: float,
#     has_sparse_grad: bool,
#     maximize: bool,
#     differentiable: bool,
#     has_complex: bool,
# ):

#     for (param, grad, state_sum, step_t) in zip(params, grads, state_sums, state_steps):
#         # update step
#         step_t += 1
#         step = _get_value(step_t)
#         grad = grad if not maximize else -grad

#         if weight_decay != 0:
#             if grad.is_sparse:
#                 raise RuntimeError(
#                     "weight_decay option is not compatible with sparse gradients"
#                 )
#             grad = grad.add(param, alpha=weight_decay)

#         clr = lr / (1 + (step - 1) * lr_decay)

#         if grad.is_sparse:
#             grad = grad.coalesce()  # the update is non-linear so indices must be unique
#             grad_indices = grad._indices()
#             grad_values = grad._values()

#             state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))
#             std = state_sum.sparse_mask(grad)
#             std_values = std._values().sqrt_().add_(eps)
#             param.add_(
#                 _make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr
#             )
#         else:
#             is_complex = torch.is_complex(param)
#             if is_complex:
#                 grad = torch.view_as_real(grad)
#                 state_sum = torch.view_as_real(state_sum)
#                 param = torch.view_as_real(param)
#             state_sum.addcmul_(grad, grad, value=1)
#             if differentiable:
#                 std = state_sum.sqrt() + eps
#             else:
#                 std = state_sum.sqrt().add_(eps)
#             param.addcdiv_(grad, std, value=-clr)
#             if is_complex:
#                 param = torch.view_as_complex(param)
#                 state_sum = torch.view_as_complex(state_sum)


def _multi_tensor_adagrad(
    params: List[Tensor],
    grads: List[Tensor],
    state_sums: List[Tensor],
    state_steps: List[Tensor],
    loss: Tensor,
    prev_loss: Tensor = None,
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    has_sparse_grad: bool,
    maximize: bool,
    differentiable: bool,
    has_complex: bool,
):
    assert not differentiable, "_foreach ops don't support autograd"

    # Foreach functions will throw errors if given empty lists
    if len(params) == 0:
        return

    delta = torch.sub(loss, prev_loss) if prev_loss is not None else loss

    grouped_tensorlists = Optimizer._group_tensors_by_device_and_dtype([params, grads, state_sums, state_steps])
    for ((device_params, device_grads, device_state_sums, device_state_steps), _) in grouped_tensorlists.values():
        # device_has_sparse_grad = has_sparse_grad and any(grad.is_sparse for grad in device_grads)

        # if device_has_sparse_grad:
        #     _single_tensor_adagrad(
        #         device_params,
        #         device_grads,
        #         device_state_sums,
        #         device_state_steps,
        #         lr=lr,
        #         weight_decay=weight_decay,
        #         lr_decay=lr_decay,
        #         eps=eps,
        #         has_sparse_grad=True,
        #         maximize=False,
        #         differentiable=differentiable,
        #         has_complex=has_complex,
        #     )
        #     continue

        # Handle complex parameters
        if has_complex:
            _view_as_real(device_params, device_grads, device_state_sums)

        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if device_state_steps[0].is_cpu:
            torch._foreach_add_(device_state_steps, torch.tensor(1.0, device='cpu'), alpha=1.0)
        else:
            torch._foreach_add_(device_state_steps, 1)

        if weight_decay != 0:
            # Re-use the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay)

        grads_norm = [torch.norm(device_grad) for device_grad in device_grads]
        grads_norm = [grad_norm / (1 + delta) for grad_norm in grads_norm]

        all_grads_sum = [torch.sum(device_state_sum) for device_state_sum in device_state_sums]

        gpi = [grad_norm / all_grad_sum if all_grad_sum.item() != 0 else grad_norm for grad_norm, all_grad_sum in zip(grads_norm, all_grads_sum)]
        gpi = torch.stack(gpi)
        gpi = torch.mean(gpi)
        gpi = torch.clamp(gpi, MIN_GPI_VALUE, MAX_GPI_VALUE)

        # Bug?
        # I think this line should be:
        # minus_clr = [-lr / (1 + (_get_value(step - 1) * lr_decay)) for step in device_state_steps]
        # But the original code is:
        # minus_clr = [(-lr) / (1 + (_get_value(step) - 1) * lr_decay) for step in device_state_steps]
        # -----
        # -lr' = -(lr / (1 + (step - 1) * lr_decay))
        minus_clr = [(-lr * (gpi)) / (1 + (_get_value(step) - 1) * lr_decay) for step in device_state_steps]

        # G += g * g
        torch._foreach_addcmul_(device_state_sums, device_grads, device_grads, value=1)

        # sqrt(G)
        std = torch._foreach_sqrt(device_state_sums)

        # sqrt(G) + eps
        torch._foreach_add_(std, eps)

        # (-lr') * g
        torch._foreach_mul_(device_grads, minus_clr)

        # param += (-lr' * g) / (sqrt(G) + eps)
        torch._foreach_addcdiv_(device_params, device_grads, std)
