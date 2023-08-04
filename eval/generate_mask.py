import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from diffractsim import MonochromaticField, ApertureFromImage, ApertureFromFunction, um, nm


def generate_mask(path, wavelength, size, nx, ny, distance):
    F = MonochromaticField(
        wavelength=wavelength * nm,
        extent_x=size * um, extent_y=size * um,
        Nx=nx, Ny=ny
    )
    F.add(ApertureFromImage(os.path.join(path, 'img.png'), image_size=(28.8 * um, 28.8 * um), simulation=F))
    F.propagate(-distance * um)

    E = F.get_field()
    amp = np.abs(E)
    phase = np.angle(E)

    # Save amplitude and phase
    amp_img = Image.fromarray(np.uint8(np.flip(amp / np.amax(amp), axis=0) * 255), 'L')
    amp_img.save(os.path.join(path, 'amp.png'))

    phase_img = Image.fromarray(np.uint8((np.flip(phase, axis=0) + np.pi) / (2 * np.pi) * 255), 'L')
    phase_img.save(os.path.join(path, 'phase.png'))


def generate_pattern(path, wavelength, size, nx, ny, distance):
    F = MonochromaticField(
        wavelength=wavelength * nm,
        extent_x=size * um, extent_y=size * um,
        Nx=nx, Ny=ny
    )
    F.add(ApertureFromImage(
        amplitude_mask_path=os.path.join(path, 'amp.png'),
        phase_mask_path=os.path.join(path, 'phase.png'),
        phase_mask_format='graymap',
        simulation=F)
    )

    F.propagate(distance * um)
    rgb = F.get_colors()
    F.plot_colors(rgb, xlim=(-18 * um, 18 * um), ylim=(-18 * um, 18 * um))


def regenerate_pattern(wavelength, size, nx, ny, distance):
    res = np.load('res/res_y.npy')

    F = MonochromaticField(
        wavelength=wavelength * nm,
        extent_x=size * um, extent_y=size * um,
        Nx=nx, Ny=ny
    )
    F.add(ApertureFromFunction(function=lambda xx, yy, wvl: res))

    F.propagate(distance * um)
    E = F.get_field()
    E = np.flipud(abs(E) ** 2)
    plt.imshow(E, cmap='hot')
    plt.show()


def draw_mask():
    # create an image
    out = Image.new('1', (50, 50), 0)

    # get a font
    fnt = ImageFont.truetype('font/calibrib.ttf', 50)
    # get a drawing context
    d = ImageDraw.Draw(out)

    # draw multiline text
    d.multiline_text((14, 3), 'L', font=fnt, fill=1)
    arr = np.array(out)

    out.show()
    out.save('img.png')


if __name__ == '__main__':
    # draw_mask()
    # generate_mask('aperture/L', 632, 36, 50, 50, distance=50)
    # generate_pattern('aperture/L', 632, 36, 50, 50, distance=50)
    regenerate_pattern(632, 36, 50, 50, distance=50)
