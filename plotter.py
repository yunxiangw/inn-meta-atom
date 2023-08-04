import numpy as np
import plotly.express as px
import plotly.graph_objects as go


class Plotter:
    def __init__(self):
        self.cov_fig = None
        self.hist_fig = None
        self.pos_fig = None

    def plot_cov_matrix(self, data):
        cov = np.cov(data, rowvar=False)
        self.cov_fig = px.imshow(cov, text_auto='.2f')

    def plot_hist(self, data):
        self.hist_fig = px.histogram(data)

    def plot_posterior(self, data):
        dim = [dict(range=[50, 350],
                    tickvals=[50, 100, 150, 200, 250, 300, 350],
                    label=f'param_{i}',
                    values=data[:, i]) for i in range(data.shape[1])]
        self.pos_fig = go.Figure(data=go.Parcoords(dimensions=dim))

    def get_cov_fig(self):
        return self.cov_fig

    def get_hist_fig(self):
        return self.hist_fig

    def get_posterior(self):
        return self.pos_fig
