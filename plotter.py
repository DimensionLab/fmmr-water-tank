import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from modulus.utils.io import InferencerPlotter


class CustomInferencerPlotter(InferencerPlotter):
    "Default plotter class for inferencer"

    def __call__(self, invar, outvar):
        "Default function for plotting inferencer data"

        # get input variables
        x, y = invar["x"][:, 0], invar["y"][:, 0]
        bounds = (x.min(), x.max(), y.min(), y.max())

        extent, outvar = self.interpolate_output(100, x, y, bounds, outvar)

        # make plots
        fs = []
        for k in outvar:
            surfcolor_z = outvar[k]
            sminz, smaxz = get_lims_colors(surfcolor_z)

            slice_z = get_the_slice(x, y, z, surfcolor_z)
            surfcolor_y = scalar_f(x,y,z)
            sminy, smaxy = get_lims_colors(surfcolor_y)
            vmin = min([sminz, sminy])
            vmax = max([smaxz, smaxy])
            slice_y = get_the_slice(x, y, z, surfcolor_y)

            fig1 = go.Figure(data=[slice_z, slice_y])
            fig1.update_layout(
                    title_text='Slices in volumetric data', 
                    title_x=0.5,
                    width=700,
                    height=700,
                    scene_zaxis_range=[-2,2], 
                    coloraxis=dict(colorscale='BrBG',
                                    colorbar_thickness=25,
                                    colorbar_len=0.75,
                                    **colorax(vmin, vmax)))            
                
            fig1.show() 

        return fs

    @staticmethod
    def interpolate_output(size, x, y, extent, *outvars):
        "Interpolates irregular points onto a mesh"

        # define mesh to interpolate onto
        xyi = np.meshgrid(
            np.linspace(extent[0], extent[1], size),
            np.linspace(extent[2], extent[3], size),
            indexing="ij",
        )

        # interpolate outvars onto mesh
        outvars_interp = []
        for outvar in outvars:
            outvar_interp = {}
            for k in outvar:
                outvar_interp[k] = scipy.interpolate.griddata(
                    (x, y), outvar[k][:, 0], tuple(xyi)
                )
            outvars_interp.append(outvar_interp)

        return [extent] + outvars_interp


# https://nbviewer.org/github/empet/Plotly-plots/blob/master/Plotly-Slice-in-volumetric-data.ipynb

import numpy as np
import plotly.graph_objects as go


def get_the_slice(x,y,z, surfacecolor):
    return go.Surface(x=x,
                      y=y,
                      z=z,
                      surfacecolor=surfacecolor,
                      coloraxis='coloraxis')


def get_lims_colors(surfacecolor):# color limits for a slice
    return np.min(surfacecolor), np.max(surfacecolor)


def colorax(vmin, vmax):
    return dict(cmin=vmin,
                cmax=vmax)


def render_img():
    x = np.linspace(-2,2, 50)
    y = np.linspace(-2,2, 50)
    x, y = np.meshgrid(x,y)
    z = np.zeros(x.shape)
    surfcolor_z = scalar_f(x,y,z)
    sminz, smaxz = get_lims_colors(surfcolor_z)

    slice_z = get_the_slice(x, y, z, surfcolor_z)
    x = np.linspace(-2,2, 50)
    z = np.linspace(-2,2, 50)
    x, z = np.meshgrid(x,y)
    y = -0.5 * np.ones(x.shape)
    surfcolor_y = scalar_f(x,y,z)
    sminy, smaxy = get_lims_colors(surfcolor_y)
    vmin = min([sminz, sminy])
    vmax = max([smaxz, smaxy])
    slice_y = get_the_slice(x, y, z, surfcolor_y)

    fig1 = go.Figure(data=[slice_z, slice_y])
    fig1.update_layout(
            title_text='Slices in volumetric data', 
            title_x=0.5,
            width=700,
            height=700,
            scene_zaxis_range=[-2,2], 
            coloraxis=dict(colorscale='BrBG',
                            colorbar_thickness=25,
                            colorbar_len=0.75,
                            **colorax(vmin, vmax)))            
        
    fig1.show()  
