class BrainNet:
    def __init__(self, model_common, model_coarse, model_fine, device='cpu'):
        self.model_common_ = model_common;
        self.model_coarse_ = model_coarse;
        self.model_fine_   = model_fine;
        self.model_common_.to(device);
        self.model_coarse_.to(device);
        self.model_fine_.to(  device);
        self.model_common_.eval();
        self.model_coarse_.eval();
        self.model_fine_.eval();
        
    def __call__(self, x ):
        common_output = self.model_common_(x.to(device))
        y_pred_fine   = model_classifier_fine(   common_output );
        y_pred_coarse = model_classifier_coarse( common_output );
        return y_pred_coarse, y_pred_fine