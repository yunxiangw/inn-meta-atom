import numpy as np
import matplotlib.pyplot as plt
from diffractsim import MonochromaticField, ApertureFromFunction, um, FourierPhaseRetrieval, Lens

""" Matplotlib Parameters """
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'Time',
    'font.size': 28,
})


class MetalensDesign:
    def __init__(self, wavelength, focal_length, radius, lattice_constant):
        self.wavelength = wavelength
        self.focal_length = focal_length
        self.radius = radius
        self.lattice_constant = lattice_constant

    def _phase_func(self, x, y):
        return 2 * np.pi / self.wavelength * (self.focal_length-np.sqrt(self.focal_length**2 + x**2 + y**2))

    def generate_phase_distribution(self):
        num = int(2 * self.radius / self.lattice_constant)
        offset = self.lattice_constant / 2
        x, y = np.linspace(-self.radius + offset, self.radius - offset, num), np.linspace(-self.radius + offset, self.radius - offset, num)
        xx, yy = np.meshgrid(x, y)

        vf = np.vectorize(self._phase_func)
        phase = vf(xx, yy) % (2 * np.pi)
        phase -= np.min(phase)

        mask = np.sqrt(xx**2 + yy**2) <= self.radius
        phase *= mask
        return phase


def propagate(wavelength, size, nx, ny, focal_length):
    F = MonochromaticField(
        wavelength=wavelength * um,
        extent_x=size * um, extent_y=size * um,
        Nx=nx, Ny=ny
    )

    def aperture_func(xx, yy, wvl):
        res = np.load('rebuild_phase_profile_3.npy')
        x, y = np.linspace(-19.8, 19.8, 100), np.linspace(-19.8, 19.8, 100)
        xx, yy = np.meshgrid(x, y)
        mask = np.where((xx**2 + yy**2) < 20**2, 1, np.zeros_like(xx))
        return mask * np.exp(1j * res)

    F.add(ApertureFromFunction(function=aperture_func))
    longitudinal_profile_rgb, longitudinal_profile_E, extent = F.get_longitudinal_profile(
        start_distance=0 * um,
        end_distance=1.5 * focal_length * um,
        steps=200
    )

    fig, ax = plt.subplots(layout='constrained', dpi=300)
    ax.imshow(np.abs(longitudinal_profile_E.T)**2, cmap='hot')
    ax.set_xticks((0, 200/3, 400/3, 200))
    ax.set_xticklabels(('$0$', '$50$', '$100$', '$150$'))
    ax.set_yticks((0, 100))
    ax.set_yticklabels(('$40$', '$0$'))
    ax.set_aspect(2/3.75)
    ax.set_xlabel('$z (\mu m)$')
    ax.set_ylabel('$x (\mu m)$')
    ax.get_xaxis().set_visible(False)
    plt.savefig('3xz.tif')


if __name__ == '__main__':
    """ Parameters (unit: um) """
    wavelength = 0.58
    focal_length = 100
    radius = 20
    lattice_constant = 0.4

    n = int(2 * radius / lattice_constant)
    propagate(
        wavelength=wavelength,
        size=2 * radius,
        nx=n,
        ny=n,
        focal_length=100)
