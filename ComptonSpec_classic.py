'''
Script for calculation of emission Compton spectrum in the classic formalism
v.1.0: plane wave, no radiation reaction, no speed up
'''

import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d

'''
Class structure:
    --Trajectory
    --Spectrum
''' 

class Trajectory:
    '''
    Charged particle class for trajectory calculation
    u0 - initial velocities
    r0 - initial coordinates
    gamma0 - initial gamma factor
    '''
    def __init__(self, u0=np.zeros(3), r0=np.zeros(3)):
        self.u0 = u0
        self.r0 = r0
        self.gamma0 = 1. / np.sqrt(1 - np.sum(u0**2))
        self.pi0 = self.gamma0 - u0[2]
    
    def calc_u(self, A, eta, integrate_u=False):
        '''
        Calculate vector part of 4-velocity(ux, uy, uz) in a given field A(Ax, Ay) on light cone 
        time grid - eta (t-z)
        '''
        u0 = self.u0
        pi0 = self.pi0
        u = np.zeros((3,eta.shape[0]))

        if not integrate_u:
            u[0] = u0[0] + A[0]
            u[1] = u0[1] + A[1]
            u[2] = u0[2] + (u0[0] * A[0] + u0[1] * A[1] + 0.5 * (A[0]**2 + A[1]**2)) / pi0
        
        return u
    
    def calc_u_x(self, A, eta, integrate_u=False):
        '''
        Calculate trajectories (vector part of 4-velocity u and coordinates r) in a given field 
        A(Ax, Ay) on light cone time grid - eta (t-z)
        
        integrate_u=False to use analytic solutions for u if the field is plane-wave
        integrate_u=True to integrate equations of motion
        '''
        
        d_eta = eta[1] - eta[0]
        pi0 = self.pi0
        r0 = self.r0
        
        u = self.calc_u(A, eta, integrate_u=integrate_u)
        r = np.zeros_like(u)
        r[0] = r0[0] + integrate.cumtrapz(u[0]/pi0, dx=d_eta, initial=0)
        r[1] = r0[1] + integrate.cumtrapz(u[1]/pi0, dx=d_eta, initial=0)
        r[2] = r0[2] + integrate.cumtrapz(u[2]/pi0, dx=d_eta, initial=0)
        return u, r
    

class Spectrum:
    '''
    Class to calculate emitted Compton spectrum from classic trajectories
    To initialize one needs to provide
    eta - time grid
    u - vector part of 4-velocity (u_x, u_y, u_z)
    r - coordinates (x, y, z)
    '''
    def __init__(self, eta, u, r):
        self.eta = eta
        self.u = u
        self.r = r
    
    @staticmethod
    def ft(samples, Fs, t0):
        """Approximate the Fourier Transform of a time-limited 
        signal by means of the discrete Fourier Transform.
        
        samples: signal values sampled at the positions t0 + n/Fs
        Fs: Sampling frequency of the signal
        t0: starting time of the sampling of the signal
        """
        f = np.linspace(-Fs/2, Fs/2, len(samples), endpoint=False)
        return np.fft.fftshift(np.fft.fft(samples)/Fs * np.exp(2j*np.pi*f*t0))
    
    @staticmethod
    def pad_with_zeros(samples, n_padded):
        n_points = len(samples)
        # Padding with zeros to the nearest power of 2
        n_padded = n_points * n_padded
        n_samples = n_points + 2*n_padded
        n_samples = 2**(int(np.log2(n_samples)))
        n_padded = (n_samples - n_points) // 2
        
        samples_padded = np.pad(samples, n_padded, 'constant')
        return samples_padded
    
    @staticmethod
    def pad_with_zeros_list(samples_list, n_padded):
        assert isinstance(samples_list, list)
        output = []
        for samples in samples_list:
            samples_padded = Spectrum.pad_with_zeros(samples, n_padded)
            output.append(samples_padded)
        return output
    
    def calc_spectrum_I_w(self, theta=np.pi, phi=0, n_t=100, n_padded=10):
        '''
        Calculate the Compton emission spectrum I(w) (FFT in the retarded time) at given angles 
        theta and phi
        n_t - the number of points at unit time interval
        n_padded - the amount of zero padding to do before FFT
        '''
        # Define necessary parameters
        eta, u, r = self.eta, self.u, self.r
        u_x_points, u_y_points, u_z_points = u[0], u[1], u[2]
        x_points, y_points, z_points = r[0], r[1], r[2]
        
        # Detector time corresponding to \eta and construction of adaptive grid in t
        t_eta_points = eta + (1 - np.cos(theta))*z_points - x_points * np.cos(phi) * np.sin(theta) - y_points * np.sin(phi) * np.sin(theta)
        t0 = -eta[0]
        t_end = t_eta_points[-1]
        n_t_points = int((t_end + t0)*n_t)
        assert n_t_points > 0, 'n_t_points should be > 0'
        t = np.linspace(-t0, t_end, n_t_points)
        
        # Interpolation of eta(t)
        eta_interp = interp1d(t_eta_points, eta, kind='cubic')
        
        # Interpolation of trajectories: u_x(eta), x(eta), ...
        u_x = interp1d(eta, u_x_points, kind='cubic')
        u_y = interp1d(eta, u_y_points, kind='cubic')
        u_z = interp1d(eta, u_z_points, kind='cubic')
        x = interp1d(eta, x_points, kind='cubic')
        y = interp1d(eta, y_points, kind='cubic')
        z = interp1d(eta, z_points, kind='cubic')
        
        # Integrand on retarded (detector) time grid: u_x(eta(t))
        u_x_ret = u_x(eta_interp(t))
        u_y_ret = u_y(eta_interp(t))
        u_z_ret = u_z(eta_interp(t))
        
        # Jacobian of time transform in retarded time
        Jacobian_ret = 1 + (1 - np.cos(theta))*u_z_ret - u_x_ret*np.cos(phi)*np.sin(theta) - u_y_ret*np.sin(phi)*np.sin(theta)
        
        # Samples of Ix integrals on detector time grid (t)
        Ix_samples = u_x_ret / Jacobian_ret
        Iy_samples = u_y_ret / Jacobian_ret
        Iz_samples = u_z_ret / Jacobian_ret
        
        # Padding with zeros        
        Ix_samples, Iy_samples, Iz_samples = Spectrum.pad_with_zeros_list([Ix_samples, Iy_samples, Iz_samples], n_padded)
        
        # Sampling frequency and frequency range
        n_t_points_padded = len(Ix_samples)
        Fs = n_t_points / (t0+t_end)
        f = np.linspace(-Fs/2, Fs/2, n_t_points_padded, endpoint=False)
        
        # New start time of sampling (due to padding)
        n_t_padded = (n_t_points_padded - n_t_points) // 2
        t0_new = t0 + n_t_padded/Fs
        
        # FFT for positive frequncies
        Ix = Spectrum.ft(Ix_samples, Fs, t0_new)[f>=0]
        Iy = Spectrum.ft(Iy_samples, Fs, t0_new)[f>=0]
        Iz = Spectrum.ft(Iz_samples, Fs, t0_new)[f>=0]
        f = f[f>=0]
        
        # Components of intensity
        I_theta = Ix * np.cos(theta) * np.cos(phi) + Iy * np.cos(theta) * np.sin(phi) - Iz * np.sin(theta)
        I_phi = Ix * np.sin(phi) - Iy * np.cos(phi)
        
        A_theta = -f / np.sqrt(2) * I_theta
        A_phi = f / np.sqrt(2) * I_phi
        
        # Frequency in normilized over laser frequency w_L, radiation intensity is given in
        # dimensionless units, one needs to multiply it on e^2 * w_L to obtain values in ergs
        w = 2 * np.pi * f
        I = 2*(np.abs(A_theta)**2 + np.abs(A_phi)**2)
        
        return I, w
    
    @staticmethod
    def interpolate_I_theta_w(I_theta_w_list, w_list, w_bound=[0.02, 3.]):
        # Define w grid (for theta=pi)
        idx = (w_list[-1] >= w_bound[0]) & (w_list[-1] <= w_bound[1])
        w_plot = w_list[-1][idx]
        n_theta, n_w = len(w_list), len(w_plot)
        I_theta_w = np.zeros((n_theta, n_w))
        I_theta_w[-1] = I_theta_w_list[-1][idx]
        # Interpolate spectrum for all angles on the same w grid
        for i in range(n_theta-1):
            I_theta_w_interpolator = interp1d(w_list[i], I_theta_w_list[i], fill_value='extrapolate')
            I_theta_w[i] = I_theta_w_interpolator(w_plot)
        return I_theta_w, w_plot
    
    @staticmethod
    def make_I_theta_w_fixed_length(I_theta_w_list, theta_list, w_list):
        n_theta = len(w_list)
        n_w = np.array([w.shape[0] for w in w_list]).min()
        I_theta_w_arr, theta_arr, w_arr = [np.zeros((n_theta, n_w)) for i in range(3)]
        for i in range(n_theta):
            w_arr[i] = w_list[i][:n_w]
            I_theta_w_arr[i] = I_theta_w_list[i][:n_w]
            theta_arr[i] = theta_list[i] * np.ones(n_w)
        return I_theta_w_arr, theta_arr, w_arr
            
    
    def calc_spectrum_I_theta_w(self, theta_arr, phi=0, n_t=100, n_padded=10,
                                interpolate_w=False, w_bound=[0.,3.], fixed_length=True):
        '''
        Calculate the Compton emission spectrum I(theta, w) (FFT in the retarded time) on given 
        angle theta grid and at given angle phi
        theta_arr - theta grid
        n_t - the number of points at unit time interval
        n_padded - the amount of zero padding to do before FFT
        interpolate_w=True if I(theta, w) needs to be on the same w grid
        w_bound=[w0,w1] - frequency bound for which interpolation takes place
        fixed_length=True - frequency grid in all directions has the same size (but the grid is different)
        
        Returns:
        interpolate_w=True --> I[n_theta, n_w] interpolated on frequency grid for theta=pi with w_bound
        interpolate_w=False:
            fixed_length=True  --> I[n_theta, n_w] where n_w is min length, frequency grids are different
            fixed_length=False --> I and w are lists with len=n_theta and frequency grids are of different
                                   length
        '''
        I_theta_w_list, w_list = [], []
        for theta in theta_arr:
            I, w = self.calc_spectrum_I_w(theta=theta, phi=phi, n_t=n_t, n_padded=n_padded)
            I_theta_w_list.append(I)
            w_list.append(w)
        
        if interpolate_w:
            I_theta_w_arr, w_arr = Spectrum.interpolate_I_theta_w(I_theta_w_list, w_list, w_bound=w_bound)
        else:
            if fixed_length:
                I_theta_w_arr, theta_arr, w_arr = Spectrum.make_I_theta_w_fixed_length(I_theta_w_list, theta_arr, w_list)
            else:
                I_theta_w_arr, w_arr = I_theta_w_list, w_list
            
        return I_theta_w_arr, theta_arr, w_arr
    
    def calc_spectrum_I_phi_theta_w(self, phi_arr, theta_arr, n_t=100, n_padded=10, w_bound=[0.02,3.]):
        '''
        Calculate the Compton emission spectrum I(phi, theta, w) (FFT in the retarded time) on given 
        angle theta grid and given angle phi grid
        '''
        I_phi_theta_w_list, w_list = [], []
        for phi in phi_arr:
            I_theta_w_arr, _, w_arr = self.calc_spectrum_I_theta_w(theta_arr, phi=phi, n_t=100,
                                                                   n_padded=10, interpolate_w=True,
                                                                   w_bound=w_bound)
            I_phi_theta_w_list.append(I_theta_w_arr)
            w_list.append(w_arr)
        I_phi_theta_w_arr, w_arr = np.array(I_phi_theta_w_list), w_list[0]
        return I_phi_theta_w_arr, phi_arr, theta_arr, w_arr
    
    @staticmethod
    def Lorentz_transform_I_theta_w(I, theta, w, gamma=1, forward=False, n_photons=False):
        '''
        On-axis Lorentz transformation for emission spectrum
        gamma - electron gamma factor in Lab frame
        forward = True --> w_L = gamma*w*(1 + beta * np.cos(theta))
        forward = False --> w_L = gamma*w*(1 - beta * np.cos(theta))

        In the reference frame where e was initially at rest
        Input:  d^2 I / dw dO - 2D array [n_theta, n_w] (emission spectrum)
                w - 2D array [n_theta, n_w] (frequencies), each angle has different frequency grid
                theta - 2D array [n_theta, n_w] (angles)

        In the Lab frame
        Output: d^2 I_L / dw dO - 2D array [n_theta, n_w] (emission spectrum)
                w_L - 2D array (frequencies)
                theta_L - 2D array (angles)    
        '''
        forward = 1 if forward else -1
        beta = np.sqrt(1. - 1. / gamma**2)
        n_theta, n_w = I.shape

        cos_theta_L = (np.cos(theta) + forward*beta) / (1 + forward*beta*np.cos(theta))
        assert np.abs(cos_theta_L).all() <= 1.

        theta_L = np.arccos(cos_theta_L)
        I_L, w_L = [np.zeros((n_theta, n_w)) for i in range(2)]

        # Perform Lorentz transformation separately for each direction (different frequency grids for
        # different directions)
        for i in range(n_theta):
            angle = 1 + forward * beta * np.cos(theta[i])
            w_L[i] = gamma * w[i] * angle
            if not n_photons:
                I_L[i] = (gamma * angle)**2 * I[i]
            else:
                I_L[i] = gamma * angle * I[i]

        return I_L, theta_L, w_L
    
    @staticmethod
    def collimate_I_theta_w(I_theta_w, theta, w, theta_col=0):
        '''
        Integrate over solid angle dO = sin(theta) dtheta dphi over theta_col.
        Works for uneven theta grids.
        '''
        idx_theta = (np.pi - theta) <= theta_col
        I_theta_w, theta = I_theta_w[idx_theta], theta[idx_theta]
        n_theta = theta.shape[0]
        I_w_collimated = np.zeros_like(w)
        for i in range(1,n_theta):
            d_theta = theta[i] - theta[i-1]
            I_w_collimated += I_theta_w[i] * np.sin(theta[i]) * d_theta
        return I_w_collimated
    
    @staticmethod
    def integrate_I_w(I_w, w, w_bound=[0.5,1.]):
        '''
        Find the number of photons in a given frequency bandwidth. 
        Works only for even frequency grids.
        '''
        idx_w = (w >= w_bound[0]) & (w <= w_bound[1])
        I_w, w = I_w[idx_w], w[idx_w]
        dw = w[1] - w[0]
        n_photons = np.sum(I_w) * dw
        return n_photons