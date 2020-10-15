import torch
import math
import numpy as np

class BasisFunctions(object):
    def __init__(self):
        pass

    def __len__(self):
        """Number of basis functions."""
        pass

    def evaluate(self, t):
        pass

    def integrate_t2_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        pass

    def integrate_t_times_psi(self, a, b):
        """Compute integral int_a^b t * psi(t)."""
        pass

    def integrate_psi(self, a, b):
        """Compute integral int_a^b psi(t)."""
        pass


class GaussianBasisFunctions(BasisFunctions):
    """
    Function phi(t)=Gaussian(t;Mu,Sigma) 
    Mu and Sigma obtained from the data (probability density function)
    self.mu = mu_j
    self.sigma = sigma_j
    """
    def __init__(self, mu, sigma):
        self.mu = mu.unsqueeze(0) #torch.Size([1, N, 2, 1])
        self.sigma = sigma.unsqueeze(0) #torch.Size([1, N, 2, 2])


    def __repr__(self):
        return f"GaussianBasisFunction(mu={self.mu}, sigma={self.sigma})"

    def __len__(self):
        """Number of basis functions."""
        #self.mu=[1,N,2,1]
        return self.mu.size(1)

    def _phi(self, t, sigma):
        sigma_inv= 1/2. * (sigma.inverse()+ torch.transpose(sigma.inverse(),-1,-2)) #to avoid numerical problems
        return 1. / (2. * math.pi * ((sigma.det().unsqueeze(2).unsqueeze(3))**(1./2.)) )* torch.exp(-.5 * torch.transpose(t,-1,-2) @ sigma_inv @ t)

    def _integrate_product_of_gaussians(self, Mu, Sigma):
        sigma = self.sigma + Sigma #torch.Size([batch, N, 2, 2])
        return self._phi(Mu - self.mu, sigma)

    def evaluate(self, t):
        return self._phi((t-self.mu), self.sigma)

    def integrate_t2_times_psi_gaussian(self, Mu, Sigma):
        """Compute integral int N(t; mu, sigma_sq) * t**2 * psi(t).
        """
        
        S_tilde = self._integrate_product_of_gaussians(Mu, Sigma)
        sigma_tilde = ((1/2. * (Sigma.inverse() + torch.transpose(Sigma.inverse(),-1,-2))) + (1/2. * (self.sigma.inverse() + torch.transpose(self.sigma.inverse(),-1,-2))))
        sigma_tilde=(1/2. * (sigma_tilde.inverse() + torch.transpose(sigma_tilde.inverse(),-1,-2)))
        mu_tilde= sigma_tilde @ ((1/2. * (Sigma.inverse() + torch.transpose(Sigma.inverse(),-1,-2))) @ Mu + (1/2. * (self.sigma.inverse() + torch.transpose(self.sigma.inverse(),-1,-2))) @ self.mu)
        
        return S_tilde * (sigma_tilde + mu_tilde @ torch.transpose(mu_tilde,-2,-1))    
    
    def integrate_t_times_psi_gaussian(self, Mu, Sigma):
        """Compute integral int N(t; Mu, Sigma) * t * psi(t).
        """
        S_tilde = self._integrate_product_of_gaussians(Mu, Sigma)
        sigma_tilde = ((1/2. * (Sigma.inverse() + torch.transpose(Sigma.inverse(),-1,-2))) + (1/2. * (self.sigma.inverse() + torch.transpose(self.sigma.inverse(),-1,-2))))
        sigma_tilde=(1/2. * (sigma_tilde.inverse() + torch.transpose(sigma_tilde.inverse(),-1,-2)))
        mu_tilde= sigma_tilde @ ((1/2. * (Sigma.inverse() + torch.transpose(Sigma.inverse(),-1,-2))) @ Mu + (1/2. * (self.sigma.inverse() + torch.transpose(self.sigma.inverse(),-1,-2))) @ self.mu)

        return S_tilde * mu_tilde

    def integrate_psi_gaussian(self, Mu, Sigma):
        """Compute integral int N(t; Mu, Sigma) * psi(t)."""
        return self._integrate_product_of_gaussians(Mu, Sigma)


    # adding this functions for 2D continuous sparsemax

    def sqrtm(self, M):
        # M is a 2x2 positive define matrix
        # M([batch, N, 2, 2])
        device=M.device
        dtype=M.dtype

        s=torch.sqrt(M[:,0,0,0]*M[:,0,1,1]-M[:,0,0,1]*M[:,0,0,1])
        t=torch.sqrt(M[:,0,0,0]+M[:,0,1,1]+2.*s)
        identity = torch.eye(2,dtype=dtype, device=device).unsqueeze(0)
        batch_identity = identity.repeat(M.size(0), 1, 1).unsqueeze(1)
        
        return (1./t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*(M+s.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*batch_identity))

    def inv(self, M):
        # to avoid numerical problems
        return (1/2. * (M.inverse()+ torch.transpose(M.inverse(),-1,-2)))

    def get_radius_parameters(self, theta, mu_tilde, Sigma_tilde):
        device=theta.device
        dtype=theta.dtype

        inv_Sigma_tilde = self.inv(Sigma_tilde)
        a = torch.tensor([[math.cos(theta)], [math.sin(theta)]], dtype=dtype, device=device)
        sigma_sq = 1./(torch.transpose(a,-1,-2) @ inv_Sigma_tilde @ a) # [batch, N, 1, 1]
        r0 = sigma_sq * torch.transpose(a,-1,-2) @ inv_Sigma_tilde @ mu_tilde # [batch, N, 1, 1]
        P = inv_Sigma_tilde - (sigma_sq * inv_Sigma_tilde @ a @ torch.transpose(a,-1,-2) @ inv_Sigma_tilde) # [batch, N, 2, 2]
        s_tilde = torch.sqrt(sigma_sq / (2. * math.pi * Sigma_tilde.det().unsqueeze(-1).unsqueeze(-1))) * torch.exp(-.5 * torch.transpose(mu_tilde,-1,-2) @ P @ mu_tilde) # [batch, N, 1, 1]
        sigma=torch.sqrt(sigma_sq) # [batch, N, 1, 1]

        return sigma, r0, s_tilde

    def norm_uni_gaussian(self, t):
        # normalized univariate gaussian (mu=0, sigma_sq=1)
        return ((1./ (math.sqrt(2.*math.pi))) * torch.exp(-(t ** 2) / 2.)) # [batch, N, 1, 1]

    def integrate_through_radius(self, theta, mu_tilde, Sigma_tilde):
        # returns f_theta*s_tilde
        # for the forward pass
        sigma, r0, s_tilde = self.get_radius_parameters(theta, mu_tilde, Sigma_tilde)

        f_theta = (self.norm_uni_gaussian((1-r0)/sigma)*(2*sigma**3 + sigma*(r0**2 +r0)) + 
            self.norm_uni_gaussian(-r0/sigma)*(-2.*sigma**3 - sigma*(r0**2 - 1)) -
            (torch.erf((1-r0)/(math.sqrt(2.)*sigma)) - torch.erf(-r0/(math.sqrt(2.)*sigma))) * (r0**3 + (3*sigma**2 - 1)*r0)/2
            )

        return (f_theta * s_tilde)

    def integrate_through_radius_t_N(self, theta, mu_tilde, Sigma_tilde, Mu, Sigma):
        device=theta.device
        dtype=theta.dtype

        a = torch.tensor([[math.cos(theta)], [math.sin(theta)]], dtype=dtype, device=device)
        sigma, r0, s_tilde = self.get_radius_parameters(theta, mu_tilde, Sigma_tilde)
        lbd = -torch.sqrt(1 / (math.pi * torch.sqrt(Sigma.det()))).unsqueeze(-1).unsqueeze(-1) # [batch, 1, 1, 1]

        const=torch.sqrt(-2. * lbd) * self.sqrtm(Sigma) @ a * sigma
        g_theta= ( (((const*r0) + (Mu*sigma)) * self.norm_uni_gaussian(-r0/sigma)) +
            - ((const*(1+r0)+(Mu*sigma)) * self.norm_uni_gaussian((1-r0)/sigma)) +
            .5 * (((torch.sqrt(-2. * lbd) * self.sqrtm(Sigma) @ a)*(sigma**2 + r0**2) + Mu*r0) * (torch.erf((1-r0)/(math.sqrt(2)*sigma))-torch.erf(-r0/(math.sqrt(2.)*sigma))))
            )

        return (g_theta * s_tilde) # [batch, N, 2, 1]

    def integrate_through_radius_ttT_N(self, theta, mu_tilde, Sigma_tilde, Mu, Sigma):
        device=theta.device
        dtype=theta.dtype

        a = torch.tensor([[math.cos(theta)], [math.sin(theta)]], dtype=dtype, device=device)
        sigma, r0, s_tilde = self.get_radius_parameters(theta, mu_tilde, Sigma_tilde)
        lbd = -torch.sqrt(1 / (math.pi * torch.sqrt(Sigma.det()))).unsqueeze(-1).unsqueeze(-1) # [batch, 1, 1, 1]
        sqrtm_Sigma=self.sqrtm(Sigma)

        A= (-2.*lbd) * sqrtm_Sigma @ a @ torch.transpose(a,-1,-2) @ torch.transpose(sqrtm_Sigma, -1, -2)
        B = torch.sqrt(-2.*lbd) * ((sqrtm_Sigma @ a @ torch.transpose(Mu,-1,-2)) + (Mu @ torch.transpose(a, -1, -2) @ torch.transpose(sqrtm_Sigma, -1, -2)) )
        C= Mu @ torch.transpose(Mu, -1, -2)
        A_l = (sigma ** 3) * A
        B_l = (sigma ** 2) * (3*r0*A + B)
        C_l = sigma * ((3*(r0**2)*A) + (2*r0*B) + C)
        D_l = ((r0**3)*A) + ((r0**2)*B) + (r0*C)

        m_theta = ( ( ((2+(-r0/sigma)**2)*A_l - (r0/sigma)*B_l + C_l) * (self.norm_uni_gaussian(-r0/sigma))) - 
            ( ( (2+((1-r0)/sigma)**2)*A_l + ((1-r0)/sigma)*B_l + C_l) * (self.norm_uni_gaussian((1-r0)/sigma))) +
            (.5 * (B_l + D_l) * (torch.erf((1-r0)/(math.sqrt(2)*sigma))-torch.erf(-r0/(math.sqrt(2.)*sigma))))
            )

        return (m_theta * s_tilde) # [batch, N, 2, 2]

    def integrate_through_radius_N(self, theta, mu_tilde, Sigma_tilde):
        sigma, r0, s_tilde = self.get_radius_parameters(theta, mu_tilde, Sigma_tilde)

        h_theta=(sigma * (self.norm_uni_gaussian(-r0/sigma) - self.norm_uni_gaussian((1-r0)/sigma))  +
            (r0/2)*(torch.erf((1-r0)/(math.sqrt(2)*sigma)) - torch.erf(-r0/(math.sqrt(2.)*sigma)))
            )

        return (h_theta * s_tilde) # [batch, N, 1, 1]

    def integrate_psi(self, Mu, Sigma):
        # returns the result for the forward pass
        # simple sum with n_points for the numerical integration
        lbd = -torch.sqrt(1. / (math.pi * torch.sqrt(Sigma.det()))).unsqueeze(-1).unsqueeze(-1) # [batch, 1, 1, 1]
        mu_tilde= (1./ torch.sqrt(-2.*lbd) * self.inv(self.sqrtm(Sigma))) @ (self.mu-Mu) # [batch, N, 2, 1]
        Sigma_tilde= 1. / (-2.*lbd) * self.inv(self.sqrtm(Sigma))@ self.sigma @ self.inv(self.sqrtm(Sigma)) # [batch, N, 2, 2]

        n_points=100 # integrate with 100 points
        values=torch.zeros(mu_tilde.size(0),mu_tilde.size(1),n_points,1).cuda()
        theta=torch.linspace(0,2.*math.pi,n_points).cuda()

        for i in range(n_points):
            values[:,:,i]= (-lbd * self.integrate_through_radius(theta[i], mu_tilde, Sigma_tilde)).squeeze(-1)

        result=(2*math.pi * torch.mean(values, dim=2)).unsqueeze(-1) 

        return result # [batch, N, 1, 1] 

    def integrate_t_times_psi(self, Mu, Sigma):
        # returns the result of the first integral for the backward pass
        # simple sum with n_points for the numerical integration
        lbd = -torch.sqrt(1 / (math.pi * torch.sqrt(Sigma.det()))).unsqueeze(-1).unsqueeze(-1) # [batch, 1, 1, 1]
        mu_tilde= (1./ torch.sqrt(-2*lbd) * self.inv(self.sqrtm(Sigma))) @ (self.mu-Mu) # [batch, N, 2, 1]
        Sigma_tilde= 1. / (-2*lbd) * self.inv(self.sqrtm(Sigma))@ self.sigma @ self.inv(self.sqrtm(Sigma)) # [batch, N, 2, 2]

        n_points=100 # integrate with 100 points
        values=torch.zeros(mu_tilde.size(0),mu_tilde.size(1),n_points,2).cuda()
        theta=torch.linspace(0,2*math.pi,n_points).cuda()

        for i in range(n_points):
            values[:,:,i]= (self.integrate_through_radius_t_N(theta[i], mu_tilde, Sigma_tilde, Mu, Sigma)).reshape([mu_tilde.size(0),mu_tilde.size(1),2])

        result=(2*math.pi * torch.mean(values, dim=2)).reshape([mu_tilde.size(0),mu_tilde.size(1),2,1])
        
        return result # [batch, N, 2, 1]
        
    def integrate_t2_times_psi(self, Mu, Sigma):
        # returns the result of the third integral for the backward pass
        # simple sum with n_points for the numerical integration
        lbd = -torch.sqrt(1 / (math.pi * torch.sqrt(Sigma.det()))).unsqueeze(-1).unsqueeze(-1) # [batch, 1, 1, 1]
        mu_tilde= (1./ torch.sqrt(-2*lbd) * self.inv(self.sqrtm(Sigma))) @ (self.mu-Mu) # [batch, N, 2, 1]
        Sigma_tilde= 1. / (-2*lbd) * self.inv(self.sqrtm(Sigma))@ self.sigma @ self.inv(self.sqrtm(Sigma)) # [batch, N, 2, 2]

        n_points=100 # integrate with 100 points
        values=torch.zeros(mu_tilde.size(0),mu_tilde.size(1),n_points,4).cuda()
        theta=torch.linspace(0,2*math.pi,n_points).cuda()

        for i in range(n_points):
            values[:,:,i]= (self.integrate_through_radius_ttT_N(theta[i], mu_tilde, Sigma_tilde, Mu, Sigma)).reshape([mu_tilde.size(0),mu_tilde.size(1),4])

        result=(2*math.pi * torch.mean(values, dim=2)).reshape([mu_tilde.size(0),mu_tilde.size(1),2,2])

        return result # [batch, N, 2, 2]

    def integrate_normal(self, Mu, Sigma):
        lbd = -torch.sqrt(1 / (math.pi * torch.sqrt(Sigma.det()))).unsqueeze(-1).unsqueeze(-1) # [batch, 1, 1, 1]
        mu_tilde= (1./ torch.sqrt(-2*lbd) * self.inv(self.sqrtm(Sigma))) @ (self.mu-Mu) # [batch, N, 2, 1]
        Sigma_tilde= 1. / (-2*lbd) * self.inv(self.sqrtm(Sigma))@ self.sigma @ self.inv(self.sqrtm(Sigma)) # [batch, N, 2, 2]

        n_points=100 # integrate with 100 points
        values=torch.zeros(mu_tilde.size(0),mu_tilde.size(1),n_points,1).cuda()
        theta=torch.linspace(0,2*math.pi,n_points).cuda()

        for i in range(n_points):
            values[:,:,i]= (self.integrate_through_radius_N(theta[i], mu_tilde, Sigma_tilde)).squeeze(-1)

        result=(2*math.pi * torch.mean(values, dim=2)).unsqueeze(-1) 

        return result # [batch, N, 1, 1]

    def area_ellipse(self, Mu, Sigma):
        lbd = -torch.sqrt(1 / (math.pi * torch.sqrt(Sigma.det()))).unsqueeze(-1).unsqueeze(-1) # [batch, 1, 1, 1]
        op= self.inv(Sigma) / (-2.*lbd)
        area = math.pi / (torch.sqrt(op.det()))
        return area # [batch,1]

    def aux(self, Mu, Sigma):
        aux=(Mu@torch.transpose(Mu,-1,-2)) + (Sigma/(self.area_ellipse(Mu, Sigma).unsqueeze(-1).unsqueeze(-1)))
        return aux # [batch,1,2,2]
