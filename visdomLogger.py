from visdom import Visdom
import numpy as np
class VisdomLogger():
    """
    Logger that uses visdom to create learning curves
    Parameters
    ----------
    - env: str, name of the visdom environment
    - log_checkpoints: bool, whether to use checkpoints or epoch averages
        for training loss
    - legend: tuple, names of the different losses that will be plotted.
    """
    def __init__(self,
                 server='http://localhost',
                 port=8097):
        if Visdom is None:
            warnings.warn("Couldn't import visdom: `pip install visdom`")
        else:
            self.viz = Visdom(server=server, port=port)
            # self.viz.delete_env()

    def deleteWindow(self, win):
        self.viz.close(win=win)
        
    def appendLine(self, name, win, X, Y, xlabel='empty', ylabel='empty'):
        if xlabel == 'empty' or ylabel == 'empty':
            self.viz.line(X=X, Y=Y, win=win, name=name, update='append', opts=dict(title="Loss"))
        else:
            self.viz.line(X=X, Y=Y, win=win, name=name, update='append', opts=dict(title="Loss", xlabel=xlabel, ylabel=ylabel, showlegend=True))

    def plotLine(self, name, win, X, Y):
        self.viz.line(X=X, Y=Y, win=win, name=name)

    def plotImage(self, image, win, title="Image", caption="Just a Image"):
        self.viz.image(image,
                     win=win,
                     opts=dict(title=title, caption=caption))

    def plotImages(self, images, win, nrow, caption="Validation Output"):
        self.viz.images(images,
                        win=win,
                        nrow=nrow,
                        opts=dict(caption=caption))

    def plot3dScatter(self, point, win):
        print("Point is", point)
        self.viz.scatter(X = point,
                        win=win,
                        opts=dict(update='update'))

if __name__ == "__main__":
    """
    testing Visdom Logger
    """
    logger = VisdomLogger()
    logger.plotLine(X=np.array([1,2,3,4,5]), Y=np.array([1.5,2.5,3.5,4.5,5.5]), win="test", name="justPlotALine")