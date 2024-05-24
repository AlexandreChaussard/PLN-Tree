from plntree.utils import model_utils
from plntree.utils import seed_all

class BaseModel:

    def __init__(self, classifier=False, seed=None):
        self.classifier_mode = classifier
        self.seed = seed
        seed_all(seed)

    def fit(self, optimizer, dataloader, n_epoch, max_grad_norm=1., verbose=10, writer=None, writer_fun=None):
        self.smart_init(dataloader)
        return model_utils.fit(self, optimizer, dataloader, n_epoch, max_grad_norm=max_grad_norm, verbose=verbose, writer=writer, writer_fun=writer_fun)

    def fit_alternate(self, optimizer_theta, optimizer_phi, dataloader, n_epoch, max_grad_norm_phi=1., max_grad_norm_theta=1., verbose=10, writer=None, writer_fun=None):
        self.smart_init(dataloader)
        return model_utils.fit_alternate(self, optimizer_theta, optimizer_phi, dataloader, n_epoch, max_grad_norm_phi, max_grad_norm_theta, verbose, writer, writer_fun)

    def smart_init(self, dataloader):
        return self