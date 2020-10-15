import torch
import torch.nn as nn

# torch.autograd.set_detect_anomaly(True)


class ContinuousSparsemaxFunction(torch.autograd.Function):

    @classmethod
    def _expectation_phi_psi(cls, ctx, Mu, Sigma):
        """Compute expectation of phi(t) * psi(t).T under N(mu, sigma_sq)."""
        num_basis = [len(basis_functions) for basis_functions in ctx.psi]
        total_basis = sum(num_basis)
        V = torch.zeros((Mu.shape[0], 6, total_basis), dtype=ctx.dtype, device=ctx.device)
        offsets = torch.cumsum(torch.IntTensor(num_basis).to(ctx.device), dim=0)
        start = 0
        for j, basis_functions in enumerate(ctx.psi):
            integral_t_times_psi=(basis_functions.integrate_t_times_psi(Mu,Sigma).squeeze(-1)).to(ctx.device)
            integral_t2_times_psi=basis_functions.integrate_t2_times_psi(Mu,Sigma).to(ctx.device)

            V[:, 0, start:offsets[j]]=integral_t_times_psi[:,:,0]
            V[:, 1, start:offsets[j]]=integral_t_times_psi[:,:,1]
            V[:, 2, start:offsets[j]]=integral_t2_times_psi[:,:,0,0]
            V[:, 3, start:offsets[j]]=integral_t2_times_psi[:,:,0,1]
            V[:, 4, start:offsets[j]]=integral_t2_times_psi[:,:,1,0]
            V[:, 5, start:offsets[j]]=integral_t2_times_psi[:,:,1,1]
            start = offsets[j]
        return V # [batch,6,N]


    @classmethod
    def _expectation_psi(cls, ctx, Mu, Sigma):
        """Compute expectation of psi under N(mu, sigma_sq)."""
        num_basis = [len(basis_functions) for basis_functions in ctx.psi]
        total_basis = sum(num_basis)
        r = torch.zeros(Mu.shape[0], total_basis, dtype=ctx.dtype, device=ctx.device)
        offsets = torch.cumsum(torch.IntTensor(num_basis).to(ctx.device), dim=0)
        start = 0
        for j, basis_functions in enumerate(ctx.psi):
            r[:, start:offsets[j]] = basis_functions.integrate_psi(Mu, Sigma).squeeze(-2).squeeze(-1)
            start = offsets[j]
        return r # [batch,N]


    @classmethod
    def _expectation_phi(cls, ctx, Mu, Sigma):
        
        num_basis = [len(basis_functions) for basis_functions in ctx.psi]
        total_basis = sum(num_basis)
        v = torch.zeros((Mu.shape[0], 6, total_basis), dtype=ctx.dtype, device=ctx.device)
        offsets = torch.cumsum(torch.IntTensor(num_basis).to(ctx.device), dim=0)
        start = 0
        
        for j, basis_functions in enumerate(ctx.psi):
            integral_normal=basis_functions.integrate_normal(Mu, Sigma).to(ctx.device) # [batch, N, 1, 1]
            aux=(basis_functions.aux(Mu, Sigma)).to(ctx.device)

            v[:, 0, start:offsets[j]]=Mu.squeeze(-1)[:,:,0] * integral_normal.squeeze(-1).squeeze(-1)
            v[:, 1, start:offsets[j]]=Mu.squeeze(-1)[:,:,1] * integral_normal.squeeze(-1).squeeze(-1)
            v[:, 2, start:offsets[j]]=(aux * integral_normal)[:,:,0,0]
            v[:, 3, start:offsets[j]]=(aux * integral_normal)[:,:,0,1]
            v[:, 4, start:offsets[j]]=(aux * integral_normal)[:,:,1,0]
            v[:, 5, start:offsets[j]]=(aux * integral_normal)[:,:,1,1]
            start = offsets[j]
        return v # [batch,6,N]


    @classmethod
    def forward(cls, ctx, theta, psi):
        # We assume a Gaussian
        # We have:
        # Mu:[batch,1,2,1] and Sigma:[batch,1,2,2]
        #theta=[(Sigma)^-1 @ Mu, -0.5*(Sigma)^-1]
        #theta: batch x 6 
        #phi(t)=[t,tt^t]
        #p(t)= Gaussian(t; Mu, Sigma)

        ctx.dtype = theta.dtype
        ctx.device = theta.device
        ctx.psi = psi

        Sigma=(-2*theta[:,2:6].view(-1,2,2))
        Sigma=(1/2. * (Sigma.inverse() + torch.transpose(Sigma.inverse(),-1,-2))).unsqueeze(1) # torch.Size([batch, 1, 2, 2])
        Mu=(Sigma @ (theta[:,0:2].view(-1,2,1)).unsqueeze(1)) # torch.Size([batch, 1, 2, 1])
        
        r=cls._expectation_psi(ctx, Mu, Sigma)
        ctx.save_for_backward(Mu, Sigma, r)
        return r # [batch, N]

    @classmethod
    def backward(cls, ctx, grad_output):
        Mu, Sigma, r = ctx.saved_tensors
        J = cls._expectation_phi_psi(ctx, Mu, Sigma) - cls._expectation_phi(ctx, Mu, Sigma) # batch,6,N
        grad_input = torch.matmul(J, grad_output.unsqueeze(2)).squeeze(2)
        return grad_input, None

class ContinuousSparsemax(nn.Module):
    def __init__(self, psi=None):
        super(ContinuousSparsemax, self).__init__()
        self.psi = psi

    def forward(self, theta):
        return ContinuousSparsemaxFunction.apply(theta, self.psi)
