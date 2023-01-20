from ultralytics import YOLO

from ultralytics.nn.tasks import ClassificationModel, DetectionModel, SegmentationModel, attempt_load_one_weight
from ultralytics.yolo.configs import get_config
from ultralytics.yolo.utils import DEFAULT_CONFIG, LOGGER, yaml_load
from ultralytics.yolo.utils.checks import check_imgsz, check_yaml
from ultralytics.yolo.utils.torch_utils import guess_task_from_head, smart_inference_mode

# Map head to model, trainer, validator, and predictor classes
MODEL_MAP = {
    "classify": [
        ClassificationModel, 'yolo.TYPE.classify.ClassificationTrainer', 'yolo.TYPE.classify.ClassificationValidator',
        'yolo.TYPE.classify.ClassificationPredictor'],
    "detect": [
        DetectionModel, 'yolo.TYPE.detect.DetectionTrainer', 'yolo.TYPE.detect.DetectionValidator',
        'yolo.TYPE.detect.DetectionPredictor'],
    "segment": [
        SegmentationModel, 'yolo.TYPE.segment.SegmentationTrainer', 'yolo.TYPE.segment.SegmentationValidator',
        'yolo.TYPE.segment.SegmentationPredictor']}

def _guess_ops_from_task(task):
        type="v8"
        model_class, train_lit, val_lit, pred_lit = MODEL_MAP[task]
        # warning: eval is unsafe. Use with caution
        trainer_class = eval(train_lit.replace("TYPE", f"{type}"))
        validator_class = eval(val_lit.replace("TYPE", f"{type}"))
        predictor_class = eval(pred_lit.replace("TYPE", f"{type}"))

        return model_class, trainer_class, validator_class, predictor_class

def mycreatemodel(cfg: str, verbose=True):
    #ultralytics/yolo/engine/model.py -> _new
    cfg = check_yaml(cfg)  # check YAML
    cfg_dict = yaml_load(cfg, append_filename=True)  # model dict
    task = guess_task_from_head(cfg_dict["head"][-1][-2])
    # ModelClass, TrainerClass, ValidatorClass, PredictorClass = \
    #         _guess_ops_from_task(task)
    ModelClass, train_lit, val_lit, pred_lit = MODEL_MAP[task]
    #ultralytics/nn/tasks.py
    model = ModelClass(cfg_dict, verbose=verbose)  # initialize
    return model

@staticmethod
def _reset_ckpt_args(args):
    args.pop("project", None)
    args.pop("name", None)
    args.pop("batch", None)
    args.pop("epochs", None)
    args.pop("cache", None)
    args.pop("save_json", None)

    # set device to '' to prevent from auto DDP usage
    args["device"] = ''

def myloadmodelweights(weights: str):
    #ultralytics/yolo/engine/model.py -> _load
    model, ckpt = attempt_load_one_weight(weights)
    ckpt_path = weights
    task = model.args["task"]
    overrides = model.args
    _reset_ckpt_args(overrides)
    ModelClass, train_lit, val_lit, pred_lit = MODEL_MAP[task]
    return model

if __name__ == '__main__':
    
    
    # cfg = "yolov8n.yaml"
    # model = mycreatemodel(cfg)

    # model = myloadmodelweights("yolov8n.pt")




    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # # Use the model
    #results = model.train(data="coco.yaml", epochs=3)  # train the model
    #results = model.val()  # evaluate model performance on the validation set
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    #success = model.export(format="onnx")  # export the model to ONNX format
    success = model.export(format="engine")