# runner.py

import os
import tqdm
import json
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from scipy.ndimage import maximum_filter

import networkx as nx
from networkx.algorithms.dag import lexicographical_topological_sort

from models.models import ASDiffusionBackbone
from utils.semantic_prototype_loader import load_task_semantic_prototypes

from datasets.gtg_dataset_loader import get_data_dict, VideoDataset

from utils.metrics import Video, Checkpoint, omission_detection
from utils.utils import draw_pred, create_image_grid

from dp.graph_utils import compute_generalized_metadag_costs, generalized_metadag2vid

from src.erm import SoftCandidateERM

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


def mode_filter(x, window_size=30):
    assert window_size >= 1, "Window size must be at least 1"
    assert isinstance(window_size, int), "Window size must be an integer"

    n = len(x)
    filtered = np.zeros_like(x, dtype=int)

    for i in range(n):
        start = max(0, i - window_size // 2 + 1)
        end = min(n, i + window_size // 2)
        filtered[i] = np.bincount(x[start:end]).argmax()

    return filtered


def create_log_folder(dirname, naming, dataset_name, ckpt_dir):
    now = datetime.now()
    current_time = now.strftime("%m_%d_%H_%M_%S")

    ckpt_dataset_dir = os.path.join(ckpt_dir, naming, dataset_name)
    runs_dataset_dir = os.path.join("runs", naming, dataset_name)

    os.makedirs(ckpt_dataset_dir, exist_ok=True)
    os.makedirs(runs_dataset_dir, exist_ok=True)

    if dirname == "debug":
        save_dir = os.path.join(ckpt_dataset_dir, "debug")
        writer_dir = os.path.join(runs_dataset_dir, "debug")
    else:
        save_dir = os.path.join(ckpt_dataset_dir, dirname + "_" + current_time)
        writer_dir = os.path.join(runs_dataset_dir, dirname + "_" + current_time)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(writer_dir, exist_ok=True)

    writer = SummaryWriter(writer_dir)
    return save_dir, writer


def segments_to_framewise(segments, action_list, length, default_cls=0):
    preds = torch.zeros((length)) + default_cls
    for j in range(len(segments)):
        st = int(segments[j, 0].item())
        ed = int(segments[j, 1].item())
        if st != ed:
            preds[st:ed] = action_list[j]

    return preds.long()


def get_datasets(all_params, num_classes, action2idx, actiontype2idx, is_eval):
    root_data_dir = all_params["root_data_dir"]
    dataset_name = all_params["dataset_name"]
    naming = all_params["naming"]

    suffix = ""
    if naming == "CaptainCook4D":
        addition_name = "Other"
        suffix = "_360p"
    elif naming == "EgoPER":
        addition_name = "Error_Addition"

    v_feature_dir = os.path.join(root_data_dir, dataset_name, all_params["v_feat_path"])
    label_dir = os.path.join(root_data_dir, dataset_name, all_params["label_path"])

    with open(os.path.join(root_data_dir, dataset_name, all_params["train_split"] + ".txt"), "r") as fp:
        lines = fp.readlines()
        train_video_list = [line.strip("\n") for line in lines]

    with open(os.path.join(root_data_dir, dataset_name, all_params["val_split"] + ".txt"), "r") as fp:
        lines = fp.readlines()
        val_video_list = [line.strip("\n") for line in lines]

    with open(os.path.join(root_data_dir, dataset_name, all_params["test_split"] + ".txt"), "r") as fp:
        lines = fp.readlines()
        test_video_list = [line.strip("\n") for line in lines]

    train_data_dict = get_data_dict(
        v_feature_dir=v_feature_dir,
        label_dir=label_dir,
        video_list=train_video_list,
        action2idx=action2idx,
        actiontype2idx=actiontype2idx,
        addition_name=addition_name,
        suffix=suffix,
    )

    val_data_dict = get_data_dict(
        v_feature_dir=v_feature_dir,
        label_dir=label_dir,
        video_list=val_video_list,
        action2idx=action2idx,
        actiontype2idx=actiontype2idx,
        addition_name=addition_name,
        suffix=suffix,
    )

    test_data_dict = get_data_dict(
        v_feature_dir=v_feature_dir,
        label_dir=label_dir,
        video_list=test_video_list,
        action2idx=action2idx,
        actiontype2idx=actiontype2idx,
        addition_name=addition_name,
        suffix=suffix,
    )

    train_dataset = VideoDataset(root_data_dir, train_data_dict, num_classes, mode="train", naming=naming, dataset_name=dataset_name)
    val_dataset = VideoDataset(root_data_dir, val_data_dict, num_classes, mode="test", naming=naming, dataset_name=dataset_name)
    test_dataset = VideoDataset(root_data_dir, test_data_dict, num_classes, mode="test", naming=naming, dataset_name=dataset_name)

    return {"train": train_dataset, "val": val_dataset, "test": test_dataset}


class Runner:
    def __init__(self, args):
        all_params = json.load(open(args.config))
        root_data_dir = all_params["root_data_dir"]
        dataset_name = all_params["dataset_name"]
        input_dim = all_params["input_dim"]

        self.root_data_dir = root_data_dir
        self.input_dim = input_dim
        self.simple_error_path = all_params.get("simple_error_path", "vc_chatgpt4omini_error_features")

        self.naming = all_params["naming"]
        self.lr = all_params["learning_rate"]
        self.weight_decay = all_params["weight_decay"]
        self.num_epochs = all_params["num_epochs"]
        self.log_freq = all_params["log_freq"]
        self.ignore_idx = all_params["ignore_idx"]
        self.batch_size = all_params["batch_size"]
        self.num_iterations = all_params["num_iterations"]
        self.ckpt_dir = all_params["ckpt_dir"]
        self.drop_base = all_params["drop_base"]
        self.is_vis = args.vis
        self.is_training = not args.eval
        self.dataset_name = dataset_name

        # visual-memory config
        self.use_visual_memory = all_params.get("use_visual_memory", False)
        self.use_semantic_memory = all_params.get("use_semantic_memory", False)

        self.semantic_short_dim = all_params.get("semantic_short_dim", 256)
        self.semantic_long_dim = all_params.get("semantic_long_dim", 384)
        self.semantic_uncertainty_dim = all_params.get("semantic_uncertainty_dim", 32)
        self.semantic_long_write_cap = all_params.get("semantic_long_write_cap", 0.2)

        self.semantic_tau_step = all_params.get("semantic_tau_step", 0.07)
        self.semantic_tau_err = all_params.get("semantic_tau_err", 0.07)
        self.semantic_rho_err = all_params.get("semantic_rho_err", 0.85)
        self.semantic_error_candidate_max_k = all_params.get("semantic_error_candidate_max_k", 5)

        self.semantic_topo_lambda_self = all_params.get("semantic_topo_lambda_self", 1.0)
        self.semantic_topo_lambda_succ = all_params.get("semantic_topo_lambda_succ", 0.8)
        self.semantic_topo_lambda_pred = all_params.get("semantic_topo_lambda_pred", 0.4)
        self.semantic_topo_lambda_total = all_params.get("semantic_topo_lambda_total", 0.5)

        self.semantic_feature_dir = all_params.get("semantic_feature_dir", "vc_normal_action_features")
        self.semantic_error_feature_dir = all_params.get(
            "semantic_error_feature_dir",
            all_params.get("simple_error_path", "vc_chatgpt4omini_error_features"),
        )
        self.semantic_feature_dim = all_params.get("semantic_feature_dim", self.input_dim)

        self.semantic_proto_payload = None
        self.erm_module = None

        self.short_dim = all_params.get("short_dim", 256)
        self.long_dim = all_params.get("long_dim", 384)
        self.fusion_dim = all_params.get("fusion_dim", 256)
        self.long_write_cap = all_params.get("long_write_cap", 0.2)
        self.fusion_dropout = all_params.get("fusion_dropout", 0.1)
        self.pretrained_backbone_ckpt = all_params.get("pretrained_backbone_ckpt", "")

        # ERM v2 config
        self.use_new_erm = all_params.get("use_new_erm", False)
        self.erm_rho = all_params.get("erm_rho", 0.85)
        self.erm_kmax_sem = all_params.get("erm_kmax_sem", 5)
        self.erm_kmax_final = all_params.get("erm_kmax_final", 6)

        self.erm_lambda_anchor = all_params.get("erm_lambda_anchor", 0.8)
        self.erm_lambda_nb = all_params.get("erm_lambda_nb", 0.3)
        self.erm_lambda_cov = all_params.get("erm_lambda_cov", 0.2)

        self.erm_lambda_vis = all_params.get("erm_lambda_vis", 0.5)
        self.erm_lambda_sem = all_params.get("erm_lambda_sem", 0.7)
        self.erm_lambda_obs = all_params.get("erm_lambda_obs", 0.3)

        self.erm_similarity_scale = all_params.get("erm_similarity_scale", 20.0)
        self.erm_smooth_window = all_params.get("erm_smooth_window", 5)

        self.erm_addition_bias = all_params.get("erm_addition_bias", -1.5)
        self.erm_lambda_add_bg = all_params.get("erm_lambda_add_bg", 2.5)
        self.erm_lambda_add_fallback = all_params.get("erm_lambda_add_fallback", 1.2)
        self.erm_lambda_add_lowconf = all_params.get("erm_lambda_add_lowconf", 1.0)
        self.erm_lambda_add_entropy = all_params.get("erm_lambda_add_entropy", 0.8)
        self.erm_lambda_add_mismatch = all_params.get("erm_lambda_add_mismatch", 2.0)
        self.erm_addition_scale = all_params.get("erm_addition_scale", 2.0)

        if self.use_visual_memory or self.use_semantic_memory:
            self.backbone_lr = all_params.get("backbone_learning_rate", 5e-5)
            self.vm_lr = all_params.get("vm_learning_rate", 1e-4)
        else:
            self.backbone_lr = self.lr
            self.vm_lr = self.lr

        with open(os.path.join(root_data_dir, "action2idx.json"), "r") as fp:
            self.action2idx = json.load(fp)[dataset_name]

        with open(os.path.join(root_data_dir, "actiontype2idx.json"), "r") as fp:
            self.actiontype2idx = json.load(fp)

        with open(os.path.join(root_data_dir, "idx2action.json"), "r") as fp:
            self.idx2action = json.load(fp)[dataset_name]

        with open(os.path.join(root_data_dir, "idx2actiontype.json"), "r") as fp:
            self.idx2actiontype = json.load(fp)

        self.num_classes = len(self.action2idx)
        self.gtg_num_classes = self.num_classes * 2
        self.bg_idx = self.action2idx["BG"]

        # error as additional class
        self.draw_idx2action = dict(self.idx2action)
        self.draw_idx2action["-1"] = "Error"

        if self.naming == "EgoPER":
            self.addition_idx = self.actiontype2idx["Error_Addition"]
        elif self.naming == "CaptainCook4D":
            self.addition_idx = self.actiontype2idx["Preparation Error"]
        self.num_types = len(self.actiontype2idx) - 1

        if "node_drop_base" in all_params:
            self.node_drop_base = all_params["node_drop_base"]
        else:
            self.node_drop_base = self.gtg_num_classes

        if "window_size" in all_params:
            self.window_size = all_params["window_size"]
        else:
            self.window_size = 30

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ASDiffusionBackbone(
            input_dim=input_dim,
            num_classes=self.gtg_num_classes,
            real_num_classes=self.num_classes,
            num_types=self.num_types,
            addition_idx=self.addition_idx,
            device=self.device,
            bg_w=all_params["background_weight"],
            use_visual_memory=self.use_visual_memory,
            use_semantic_memory=self.use_semantic_memory,
            uncertainty_dim=self.semantic_uncertainty_dim,
            tau_step=self.semantic_tau_step,
            tau_err=self.semantic_tau_err,
            rho_err=self.semantic_rho_err,
            error_candidate_max_k=self.semantic_error_candidate_max_k,
            topo_lambda_self=self.semantic_topo_lambda_self,
            topo_lambda_succ=self.semantic_topo_lambda_succ,
            topo_lambda_pred=self.semantic_topo_lambda_pred,
            topo_lambda_total=self.semantic_topo_lambda_total,
            short_dim=self.short_dim,
            long_dim=self.long_dim,
            fusion_dim=self.fusion_dim,
            long_write_cap=self.long_write_cap,
            fusion_dropout=self.fusion_dropout,
        )
        self.model.to(self.device)

        if self.use_semantic_memory:
            self.semantic_proto_payload = load_task_semantic_prototypes(
                root_data_dir=self.root_data_dir,
                dataset_name=self.dataset_name,
                feature_dim=self.semantic_feature_dim,
                normal_dir_name=self.semantic_feature_dir,
                error_dir_name=self.semantic_error_feature_dir,
            )

            self.model.configure_semantic_prototypes(
                step_prototypes=self.semantic_proto_payload["step_prototypes"].to(self.device),
                error_prototypes=self.semantic_proto_payload["error_prototypes"].to(self.device),
                step_node_ids=self.semantic_proto_payload["step_node_ids"],
                predecessor_edges=self.semantic_proto_payload["predecessor_edges"],
            )

            if self.use_new_erm:
                self.erm_module = SoftCandidateERM(
                    bg_idx=self.bg_idx,
                    addition_idx=self.addition_idx,
                    num_types=self.num_types,
                    step_prototypes=self.semantic_proto_payload["step_prototypes"],
                    error_prototypes=self.semantic_proto_payload["error_prototypes"],
                    step_node_ids=self.semantic_proto_payload["step_node_ids"],
                    type_ids=self.semantic_proto_payload["type_ids"],
                    rho=self.erm_rho,
                    kmax_sem=self.erm_kmax_sem,
                    kmax_final=self.erm_kmax_final,
                    lambda_anchor=self.erm_lambda_anchor,
                    lambda_nb=self.erm_lambda_nb,
                    lambda_cov=self.erm_lambda_cov,
                    lambda_vis=self.erm_lambda_vis,
                    lambda_sem=self.erm_lambda_sem,
                    lambda_obs=self.erm_lambda_obs,
                    similarity_scale=self.erm_similarity_scale,
                    smooth_window=self.erm_smooth_window,
                    addition_bias=self.erm_addition_bias,
                    lambda_add_bg=self.erm_lambda_add_bg,
                    lambda_add_fallback=self.erm_lambda_add_fallback,
                    lambda_add_lowconf=self.erm_lambda_add_lowconf,
                    lambda_add_entropy=self.erm_lambda_add_entropy,
                    lambda_add_mismatch=self.erm_lambda_add_mismatch,
                    addition_scale=self.erm_addition_scale,
                ).to(self.device)

            print("[semantic] task_dir:", self.semantic_proto_payload["task_dir"])
            print("[semantic] normal_dir:", self.semantic_proto_payload["normal_dir"])
            print("[semantic] error_dir:", self.semantic_proto_payload["error_dir"])
            print("[semantic] step_prototypes:", tuple(self.semantic_proto_payload["step_prototypes"].shape))
            print("[semantic] error_prototypes:", tuple(self.semantic_proto_payload["error_prototypes"].shape))
            print("[semantic] num_error_types:", self.semantic_proto_payload["num_error_types"])
            print("[semantic] missing_error_pairs:", len(self.semantic_proto_payload["missing_error_pairs"]))

        if self.use_new_erm and not self.use_semantic_memory:
            raise ValueError("use_new_erm=True requires use_semantic_memory=True")

        if (not args.eval) and (self.use_visual_memory or self.use_semantic_memory) and self.pretrained_backbone_ckpt:
            self._load_pretrained_backbone_if_needed(self.pretrained_backbone_ckpt)

        if self.use_visual_memory or self.use_semantic_memory:
            backbone_params = [p for p in self.model.backbone_parameters() if p.requires_grad]
            vm_params = [p for p in self.model.visual_memory_parameters() if p.requires_grad]

            self.opt = optim.Adam(
                [
                    {"params": backbone_params, "lr": self.backbone_lr},
                    {"params": vm_params, "lr": self.vm_lr},
                ],
                weight_decay=self.weight_decay,
            )
        else:
            self.opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        dataset_dict = get_datasets(all_params, self.num_classes, self.action2idx, self.actiontype2idx, args.eval)

        if args.eval:
            self.test_loader = torch.utils.data.DataLoader(dataset_dict["test"], batch_size=1, shuffle=False, num_workers=1)
            self.val_loader = torch.utils.data.DataLoader(dataset_dict["val"], batch_size=1, shuffle=False, num_workers=1)
            self.train_loader = torch.utils.data.DataLoader(dataset_dict["train"], batch_size=1, shuffle=False, num_workers=1)

            load_dir_name = args.load_dir if args.load_dir is not None else args.dir
            save_dir_name = args.save_dir if args.save_dir is not None else load_dir_name

            self.load_dir = os.path.join(
                self.ckpt_dir,
                self.naming,
                dataset_name,
                load_dir_name,
                "best_checkpoint.pth",
            )

            self.save_dir = os.path.join(
                self.ckpt_dir,
                self.naming,
                dataset_name,
                save_dir_name,
            )
            os.makedirs(self.save_dir, exist_ok=True)

            self.writer = None
        else:
            self.test_loader = torch.utils.data.DataLoader(dataset_dict["test"], batch_size=1, shuffle=False, num_workers=1)
            self.val_loader = torch.utils.data.DataLoader(dataset_dict["val"], batch_size=1, shuffle=False, num_workers=1)
            self.train_loader = torch.utils.data.DataLoader(dataset_dict["train"], batch_size=1, shuffle=True, num_workers=1)
            self.save_dir, self.writer = create_log_folder(args.dir, self.naming, dataset_name, self.ckpt_dir)

        self.G = dataset_dict["train"].G

        ############################################
        # generate error/normal description features
        ############################################
        with open(os.path.join(root_data_dir, all_params["dataset_name"], all_params["simple_error_filename"] + ".txt"), "r") as fp:
            self.error_list = fp.readlines()

        with open(os.path.join(root_data_dir, all_params["dataset_name"], "normal_actions.txt"), "r") as fp:
            self.normal_action_list = fp.readlines()

        self.action_error_dict = {}
        for error in self.error_list:
            name, des = error.split(" ")
            _, action, _, action_type, err_idx = name.split("_")
            action = int(action)
            action_type = int(action_type)

            feature = np.load(os.path.join(root_data_dir, all_params["dataset_name"], all_params["simple_error_path"], name + ".npy"))
            feature = torch.from_numpy(feature).float()

            if action not in self.action_error_dict:
                self.action_error_dict[action] = {}

            if action_type not in self.action_error_dict[action]:
                self.action_error_dict[action][action_type] = []

            self.action_error_dict[action][action_type].append(feature)

        self.merge_action_error_dict = {}
        for k, v in self.action_error_dict.items():
            self.merge_action_error_dict[k] = {}
            for s_k, s_v in v.items():
                self.merge_action_error_dict[k][s_k] = torch.stack(s_v).mean(0)

        self.normal_action_dict = {}
        for normal_action in self.normal_action_list:
            name, des = normal_action.split(" ")
            _, action = name.split("_")

            feature = np.load(os.path.join(root_data_dir, all_params["dataset_name"], "vc_normal_action_features", name + ".npy"))
            feature = torch.from_numpy(feature).float()

            if action not in self.normal_action_dict:
                self.normal_action_dict[int(action)] = feature

        ################################################
        # for evaluation, ignore non-existing action type
        ################################################
        if self.naming == "EgoPER":
            self.ignore_actions = []
        elif self.naming == "CaptainCook4D":
            self.ignore_actions = ["Normal", "Other"]

        exist_type = []
        for video_idx, data in enumerate(self.test_loader):
            v_feature, label, type_label, video = data
            type_label = type_label.squeeze(0)
            for idx in range(self.num_types):
                action_type = idx + 1
                if action_type not in exist_type and (type_label == action_type).sum() > 0:
                    exist_type.append(action_type)

        for action_type in range(self.num_types + 1):
            if action_type not in exist_type and self.idx2actiontype[str(action_type)] not in self.ignore_actions:
                self.ignore_actions.append(self.idx2actiontype[str(action_type)])

        print("Use visual memory:", self.use_visual_memory)
        print("Use semantic memory:", self.use_semantic_memory)
        print("Use new ERM:", self.use_new_erm)
        if self.use_visual_memory or self.use_semantic_memory:
            print(f"Backbone LR: {self.backbone_lr}, VM LR: {self.vm_lr}")
        print("Ignore specific or non-exsting action type for error recognition:", self.ignore_actions)

    def _load_pretrained_backbone_if_needed(self, ckpt_path):
        if ckpt_path is None or ckpt_path == "":
            print("No pretrained backbone checkpoint specified.")
            return

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]

        model_state = self.model.state_dict()
        loaded_keys = []
        skipped_keys = []

        for k, v in state_dict.items():
            if k.startswith("conv_in.") or k.startswith("module."):
                if k in model_state and model_state[k].shape == v.shape:
                    model_state[k] = v
                    loaded_keys.append(k)
                else:
                    skipped_keys.append(k)

        self.model.load_state_dict(model_state, strict=False)

        print(f"Loaded pretrained backbone from: {ckpt_path}")
        print(f"Loaded trunk params: {len(loaded_keys)}")
        if len(skipped_keys) > 0:
            print(f"Skipped trunk params (shape/key mismatch): {len(skipped_keys)}")

    def from_framewise_to_steps(self, labels, ignore_bg=True):
        pre_label = None

        steps = []
        timestamps = []
        st, ed = 0, 0

        for i in range(len(labels)):
            label = labels[i].unsqueeze(0)

            if pre_label is None:
                pre_label = label

            if pre_label != label:
                if (not ignore_bg) or (ignore_bg and pre_label != self.bg_idx):
                    steps.append(pre_label)
                    timestamps.append([st, ed])
                st = ed
                pre_label = label

            ed += 1

        if pre_label is not None:
            if (not ignore_bg) or (ignore_bg and pre_label != self.bg_idx):
                steps.append(pre_label)
                timestamps.append([st, ed])

        if len(steps) == 0:
            empty_steps = torch.empty((0,), dtype=torch.long)
            empty_timestamps = torch.empty((0, 2), dtype=torch.float32)
            return empty_steps, empty_timestamps

        return torch.cat(steps, 0).long(), torch.tensor(timestamps).float()

    def get_data_sample(self):
        v_feature, label, type_label, video = next(iter(self.train_loader))
        v_feature = v_feature.to(self.device)
        label = label.to(self.device)
        type_label = type_label.to(self.device)
        return v_feature, label, type_label, video

    def relabel_bg(self, steps):
        for i in range(len(steps) - 1):
            if steps[i] == self.bg_idx:
                if steps[i + 1] == self.bg_idx:
                    print("action after bg should a normal action")
                    exit(0)
                else:
                    steps[i] = self.num_classes - 1 + steps[i + 1]
        return steps

    def _temporal_delta_mean(self, seq):
        if seq.size(1) <= 1:
            return 0.0
        return (seq[:, 1:] - seq[:, :-1]).norm(dim=-1).mean().item()

    def _compute_memory_log_dict(self, aux):
        if (not (self.use_visual_memory or self.use_semantic_memory)) or ("base_seq" not in aux):
            return {}

        scorer = self.model.visual_memory_scorer

        with torch.no_grad():
            base_seq = aux["base_seq"].detach()
            fused_seq = aux["fused_seq"].detach()

            logs = {}

            base_proj = scorer.base_fuse_proj(base_seq)
            fusion_delta = fused_seq - base_proj
            fusion_ratio = (
                fusion_delta.norm(dim=-1).mean() / (base_proj.norm(dim=-1).mean() + 1e-8)
            ).item()
            logs["MEM/fusion_ratio"] = fusion_ratio

            if "short_memory_seq" in aux and "long_memory_seq" in aux and "long_write_gate_seq" in aux:
                short_seq = aux["short_memory_seq"].detach()
                long_seq = aux["long_memory_seq"].detach()
                gate_seq = aux["long_write_gate_seq"].detach()

                logs["MEM/short_norm"] = short_seq.norm(dim=-1).mean().item()
                logs["MEM/long_norm"] = long_seq.norm(dim=-1).mean().item()
                logs["MEM/delta_short"] = self._temporal_delta_mean(short_seq)
                logs["MEM/delta_long"] = self._temporal_delta_mean(long_seq)
                logs["MEM/gate_mean"] = gate_seq.mean().item()

                near_cap_threshold = max(getattr(self, "long_write_cap", 0.2) - 0.02, 0.0)
                logs["MEM/gate_near_zero_ratio"] = (gate_seq < 0.02).float().mean().item()
                logs["MEM/gate_near_cap_ratio"] = (gate_seq > near_cap_threshold).float().mean().item()

                if hasattr(scorer, "short_fuse_proj") and hasattr(scorer, "long_fuse_proj"):
                    short_proj = scorer.short_fuse_proj(short_seq)
                    long_proj = scorer.long_fuse_proj(long_seq)
                    logs["MEM/cos_base_short"] = F.cosine_similarity(base_proj, short_proj, dim=-1).mean().item()
                    logs["MEM/cos_base_long"] = F.cosine_similarity(base_proj, long_proj, dim=-1).mean().item()
                    logs["MEM/cos_short_long"] = F.cosine_similarity(short_proj, long_proj, dim=-1).mean().item()

            if self.use_semantic_memory:
                if "sem_short_seq" in aux:
                    sem_short_seq = aux["sem_short_seq"].detach()
                    logs["SEM/short_norm"] = sem_short_seq.norm(dim=-1).mean().item()
                    logs["SEM/delta_short"] = self._temporal_delta_mean(sem_short_seq)

                if "sem_long_seq" in aux:
                    sem_long_seq = aux["sem_long_seq"].detach()
                    logs["SEM/long_norm"] = sem_long_seq.norm(dim=-1).mean().item()
                    logs["SEM/delta_long"] = self._temporal_delta_mean(sem_long_seq)

                if "sem_long_gate_seq" in aux:
                    sem_long_gate_seq = aux["sem_long_gate_seq"].detach()
                    logs["SEM/gate_mean"] = sem_long_gate_seq.mean().item()

                    near_cap_threshold = max(getattr(self, "semantic_long_write_cap", 0.2) - 0.02, 0.0)
                    logs["SEM/gate_near_zero_ratio"] = (sem_long_gate_seq < 0.02).float().mean().item()
                    logs["SEM/gate_near_cap_ratio"] = (sem_long_gate_seq > near_cap_threshold).float().mean().item()

                if "coverage_trace_seq" in aux:
                    coverage_seq = aux["coverage_trace_seq"].detach()
                    logs["SEM/coverage_mean"] = coverage_seq.mean().item()
                    logs["SEM/coverage_peak"] = coverage_seq.max(dim=-1).values.mean().item()

                if "uncertainty_trace_seq" in aux:
                    uncertainty_seq = aux["uncertainty_trace_seq"].detach()
                    logs["SEM/uncertainty_mean"] = uncertainty_seq.mean().item()
                    logs["SEM/uncertainty_norm"] = uncertainty_seq.norm(dim=-1).mean().item()

                if "semantic_fuse_gate_seq" in aux:
                    fuse_gate_seq = aux["semantic_fuse_gate_seq"].detach()
                    logs["SEM/fuse_gate_mean"] = fuse_gate_seq.mean().item()
                    logs["SEM/fuse_gate_low_ratio"] = (fuse_gate_seq < 0.2).float().mean().item()
                    logs["SEM/fuse_gate_high_ratio"] = (fuse_gate_seq > 0.8).float().mean().item()

                if "proto_gate" in aux:
                    proto_gate = aux["proto_gate"].detach()
                    logs["SEM/proto_gate_mean"] = proto_gate.mean().item()
                    logs["SEM/proto_gate_low_ratio"] = (proto_gate < 0.2).float().mean().item()
                    logs["SEM/proto_gate_high_ratio"] = (proto_gate > 0.8).float().mean().item()

                if "step_posteriors" in aux:
                    alpha = aux["step_posteriors"].detach()
                    alpha_entropy = -(alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum(dim=-1)
                    logs["SEM/alpha_entropy"] = alpha_entropy.mean().item()
                    logs["SEM/alpha_max"] = alpha.max(dim=-1).values.mean().item()

                if "error_posteriors" in aux:
                    gamma = aux["error_posteriors"].detach()
                    error_mass = gamma.sum(dim=(-1, -2))
                    logs["SEM/error_mass_mean"] = error_mass.mean().item()

                if "main_logits" in aux and "final_logits" in aux:
                    main_logits = aux["main_logits"].detach()
                    final_logits = aux["final_logits"].detach()
                    proto_boost = final_logits - main_logits
                    logs["SEM/proto_boost_abs_mean"] = proto_boost.abs().mean().item()

            return logs

    def _flush_memory_logs(self, mem_log_buffer, global_step):
        if self.writer is None or len(mem_log_buffer) == 0:
            return

        keys = mem_log_buffer[0].keys()
        for k in keys:
            mean_v = float(np.mean([item[k] for item in mem_log_buffer]))
            self.writer.add_scalar(k, mean_v, global_step)

    def compute_per_out_log(self, joint_per_metrics, use_ignore=False, mode="as"):
        per_out_log = []
        avg_f1_thresholds = 0
        col1 = 10 if mode == "as" else 20

        for t, _ in joint_per_metrics.items():
            f1_list = []
            per_out_log.append(f"|{'Action':^{col1}}|{'Precision@%s' % t:^{15}}|{'Recall@%s' % t:^{15}}|{'F1@%s' % t:^{10}}|\n")

            total_tp, total_fp, total_fn = 0, 0, 0
            theta = 0

            for action, tp_fp_fn in joint_per_metrics[t].items():
                tp, fp, fn = 0, 0, 0
                if mode == "er":
                    action = self.idx2actiontype[str(action)]

                for i in range(len(tp_fp_fn[0])):
                    tp += tp_fp_fn[0][i]
                    fp += tp_fp_fn[1][i]
                    fn += tp_fp_fn[2][i]
                    if action not in self.ignore_actions:
                        total_tp += tp_fp_fn[0][i]
                        total_fp += tp_fp_fn[1][i]
                        total_fn += tp_fp_fn[2][i]

                p = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
                if np.isnan(p):
                    p = 0.0
                if np.isnan(r):
                    r = 0.0
                if p + r == 0:
                    f1 = 0.0
                else:
                    f1 = 2.0 * (p * r) / (p + r)

                f1 = np.nan_to_num(f1)
                p = p * 100
                r = r * 100
                f1 = f1 * 100

                if mode == "ed":
                    action = "Error" if action == 1 else "Normal"

                out_log = f"|{action:^{col1}}|{p:^{15}.1f}|{r:^{15}.1f}|{f1:^{10}.1f}|\n"
                per_out_log.append(out_log)

                if f1 != 0 and action not in self.ignore_actions:
                    theta += 1

                if not use_ignore or action not in self.ignore_actions:
                    f1_list.append(f1)

            per_out_log.append("\n")
            if use_ignore:
                per_out_log.append("Ignore action: ")
                for exclude_action in self.ignore_actions:
                    per_out_log.append(exclude_action + ", ")
            per_out_log.append("\n")
            per_out_log.append("Avg F1@%s: %.1f\n\n" % (t, np.array(f1_list).mean()))

            avg_f1_thresholds += np.array(f1_list).mean()

            if mode == "er":
                total_p = total_tp / float(total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                total_r = total_tp / float(total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                if np.isnan(total_p):
                    total_p = 0.0
                if np.isnan(total_r):
                    total_r = 0.0
                if total_p + total_r == 0:
                    total_f1 = 0.0
                else:
                    total_f1 = 2.0 * (total_p * total_r) / (total_p + total_r)

                total_f1 = np.nan_to_num(total_f1)
                total_p = total_p * 100
                total_r = total_r * 100
                total_f1 = total_f1 * 100

                eacc = theta / (len(self.actiontype2idx) - len(self.ignore_actions))
                total_w_f1 = total_f1 * eacc
                per_out_log.append(f"|{'All Precision':^{15}}|{'All Recall':^{15}}|{'All F1':^{15}}|{'All w-F1@%s' % t:^{15}}|{'EAcc':^{15}}|\n")
                per_out_log.append(f"|{total_p:^{15}.1f}|{total_r:^{15}.1f}|{total_f1:^{15}.1f}|{total_w_f1:^{15}.1f}|{eacc * 100:^{15}.1f}|\n\n")

        return per_out_log, avg_f1_thresholds / len(joint_per_metrics)

    def erm(self, pred, no_drop_pred, feature):
        cosine_similarity = nn.CosineSimilarity()

        type_pred = np.zeros((len(pred)))
        type_prob = np.zeros((self.num_types + 1, len(pred)))

        target_area = pred == -1

        for i in range(len(target_area)):
            if target_area[i]:
                target_action = no_drop_pred[i]
                target_frame_feature = feature[i]
                best_sim = None
                target_type = 0

                if target_action == self.bg_idx:
                    type_prob[self.addition_idx][i] = 1.0
                else:
                    for k, v in self.merge_action_error_dict[target_action].items():
                        sim = torch.sigmoid((v * self.normal_action_dict[target_action] * target_frame_feature).sum() * 200)
                        if best_sim is None or sim > best_sim:
                            best_sim = sim
                            target_type = k
                    type_prob[target_type][i] = best_sim
            else:
                type_prob[0][i] = 1.0

        smoothed_output = np.zeros_like(type_prob)
        for c in range(type_prob.shape[0]):
            smoothed_output[c] = maximum_filter(type_prob[c], size=self.window_size)
        type_prob = smoothed_output

        type_pred = np.argmax(type_prob, 0)
        error_pred = np.copy(type_pred)
        error_pred[error_pred > 0] = 1

        return pred, type_pred, error_pred

    def _build_new_erm_inputs(
        self,
        pred,
        no_drop_pred_raw,
        no_drop_meta,
        frame_features,
        aux,
        video_id,
    ):
        if aux is None:
            raise RuntimeError("ERM v2 requires forward_with_aux outputs")

        if "step_posteriors" not in aux:
            raise RuntimeError("ERM v2 requires semantic aux: step_posteriors")

        semantic_obs_seq = aux.get("semantic_obs_seq", None)
        if semantic_obs_seq is None and ("step_sem_obs" in aux) and ("error_sem_obs" in aux):
            semantic_obs_seq = 0.5 * (aux["step_sem_obs"] + aux["error_sem_obs"])

        def _squeeze_or_none(x):
            if x is None:
                return None
            return x.squeeze(0)

        erm_inputs = {
            "pred": pred,
            "no_drop_pred": no_drop_pred_raw,
            "no_drop_meta": no_drop_meta,
            "frame_features": frame_features.squeeze(0).permute(1, 0),  # [T, D]
            "vis_short_seq": _squeeze_or_none(aux.get("short_memory_seq", None)),
            "sem_short_seq": _squeeze_or_none(aux.get("sem_short_seq", None)),
            "semantic_obs_seq": _squeeze_or_none(semantic_obs_seq),
            "step_posteriors": _squeeze_or_none(aux.get("step_posteriors", None)),
            "coverage_trace_seq": _squeeze_or_none(aux.get("coverage_trace_seq", None)),
            "uncertainty_trace_seq": _squeeze_or_none(aux.get("uncertainty_trace_seq", None)),
            "graph": self.G.graph_info["graph"],
            "video_id": video_id,
        }
        return erm_inputs

    def _compute_erm_log_dict(self, erm_aux):
        if erm_aux is None:
            return {}

        logs = {}

        with torch.no_grad():
            candidate_count = torch.as_tensor(erm_aux["candidate_count_seq"]).float()
            err_mask = candidate_count > 0

            if err_mask.sum().item() == 0:
                return {}

            candidate_weights = torch.as_tensor(erm_aux["candidate_weights_seq"]).float()
            candidate_flags = torch.as_tensor(erm_aux["candidate_flags_seq"]).long()
            anchor_fallback = torch.as_tensor(erm_aux["anchor_fallback_seq"]).float()
            q_component_norms = torch.as_tensor(erm_aux["q_component_norms"]).float()
            joint_scores = torch.as_tensor(erm_aux["joint_scores_seq"]).float()
            aggregated_scores = torch.as_tensor(erm_aux["aggregated_scores_seq"]).float()
            smoothed_scores = torch.as_tensor(erm_aux["smoothed_scores_seq"]).float()
            addition_scores = torch.as_tensor(erm_aux.get("addition_score_seq", torch.zeros(candidate_count.shape[0]))).float()

            if aggregated_scores.ndim == 2 and aggregated_scores.shape[0] != candidate_count.shape[0]:
                aggregated_scores = aggregated_scores.transpose(0, 1)

            if smoothed_scores.ndim == 2 and smoothed_scores.shape[0] != candidate_count.shape[0]:
                smoothed_scores = smoothed_scores.transpose(0, 1)

            valid_counts = candidate_count[err_mask]
            logs["ERM/cand_size_mean"] = valid_counts.mean().item()
            logs["ERM/cand_size_std"] = (
                valid_counts.std(unbiased=False).item() if valid_counts.numel() > 1 else 0.0
            )

            anchor_mask = (candidate_flags & 1) > 0
            semantic_mask = (candidate_flags & 2) > 0
            topo_mask = (candidate_flags & 4) > 0

            anchor_weight = (candidate_weights * anchor_mask.float()).sum(dim=-1)
            logs["ERM/anchor_weight_mean"] = anchor_weight[err_mask].mean().item()

            top1_idx = candidate_weights.argmax(dim=-1, keepdim=True)
            anchor_is_top1 = anchor_mask.gather(dim=1, index=top1_idx).squeeze(1).float()
            logs["ERM/anchor_is_top1_ratio"] = anchor_is_top1[err_mask].mean().item()

            has_topo_neighbor = topo_mask.any(dim=-1).float()
            logs["ERM/topo_neighbor_in_cand_ratio"] = has_topo_neighbor[err_mask].mean().item()

            has_semantic_candidate = semantic_mask.any(dim=-1).float()
            logs["ERM/semantic_candidate_present_ratio"] = has_semantic_candidate[err_mask].mean().item()

            cand_entropy = -(
                candidate_weights.clamp_min(1e-8) * candidate_weights.clamp_min(1e-8).log()
            ).sum(dim=-1)
            logs["ERM/cand_entropy_mean"] = cand_entropy[err_mask].mean().item()

            logs["ERM/fallback_ratio"] = anchor_fallback[err_mask].mean().item()

            logs["ERM/q_frame_contrib_norm"] = q_component_norms[err_mask, 0].mean().item()
            logs["ERM/q_vis_contrib_norm"] = q_component_norms[err_mask, 1].mean().item()
            logs["ERM/q_sem_contrib_norm"] = q_component_norms[err_mask, 2].mean().item()
            logs["ERM/q_obs_contrib_norm"] = q_component_norms[err_mask, 3].mean().item()
            logs["ERM/q_norm"] = q_component_norms[err_mask, 4].mean().item()
            logs["ERM/sem_conf_mean"] = q_component_norms[err_mask, 5].mean().item()

            logs["ERM/joint_score_mean"] = joint_scores[err_mask].mean().item()
            logs["ERM/joint_score_std"] = joint_scores[err_mask].std(unbiased=False).item()
            logs["ERM/addition_score_mean"] = addition_scores[err_mask].mean().item()

            err_scores_before = aggregated_scores[err_mask, 1:]
            err_scores_after = smoothed_scores[err_mask, 1:]

            if err_scores_before.numel() > 0:
                logs["ERM/agg_score_before_smooth_mean"] = err_scores_before.mean().item()
                logs["ERM/agg_score_after_smooth_mean"] = err_scores_after.mean().item()

                top2 = torch.topk(
                    err_scores_after,
                    k=min(2, err_scores_after.shape[-1]),
                    dim=-1,
                ).values
                if top2.shape[-1] == 2:
                    margin = top2[:, 0] - top2[:, 1]
                else:
                    margin = top2[:, 0]
                logs["ERM/top1_top2_margin_mean"] = margin.mean().item()

                pred_dist = torch.softmax(err_scores_after, dim=-1)
                pred_entropy = -(
                    pred_dist.clamp_min(1e-8) * pred_dist.clamp_min(1e-8).log()
                ).sum(dim=-1)
                logs["ERM/pred_error_type_entropy"] = pred_entropy.mean().item()

                smooth_change = (err_scores_after - err_scores_before).abs().sum(dim=-1)
                logs["ERM/smooth_change_ratio"] = (smooth_change > 1e-6).float().mean().item()

        return logs

    def _dump_erm_debug_json(self, video_id, erm_aux):
        if erm_aux is None:
            return
        if not self.use_new_erm:
            return

        out_dir = os.path.join(self.save_dir, "output", "erm_debug")
        os.makedirs(out_dir, exist_ok=True)

        payload = {}
        for k, v in erm_aux.items():
            if isinstance(v, torch.Tensor):
                payload[k] = v.detach().cpu().tolist()
            elif isinstance(v, np.ndarray):
                payload[k] = v.tolist()
            else:
                payload[k] = v

        with open(os.path.join(out_dir, f"{video_id}.json"), "w") as fp:
            json.dump(payload, fp)

    def train(self):
        global_step = 0
        best_score = 0.0

        for epoch in range(self.num_epochs):
            samples = []
            mem_log_buffer = []

            for idx in range(self.num_iterations):
                feature, label, type_label, video = self.get_data_sample()

                label = label.squeeze(0).long().cpu()
                type_label = type_label.squeeze(0).long().cpu()

                if self.use_visual_memory or self.use_semantic_memory:
                    action_logits, frame_features, aux = self.model.forward_with_aux(feature.permute(0, 2, 1), label)
                    mem_log_buffer.append(self._compute_memory_log_dict(aux))
                else:
                    action_logits, frame_features = self.model(feature.permute(0, 2, 1), label)

                steps, timestamps = self.from_framewise_to_steps(label, ignore_bg=False)

                new_steps = self.relabel_bg(steps.clone())
                new_label = segments_to_framewise(timestamps, new_steps, feature.size(2))

                sample = {}
                sample["action_logits"] = action_logits.squeeze(0)
                sample["frame_features"] = frame_features.squeeze(0).permute(1, 0)
                sample["org_frame_features"] = feature.squeeze(0).permute(1, 0)
                sample["normal_action_features"] = self.normal_action_dict
                sample["framewise_labels"] = new_label
                sample["framewise_type_labels"] = type_label
                sample["video_id"] = video
                sample["metagraph"] = self.G.graph_info["metagraph"]
                sample["gmetagraph"] = self.G.graph_info["gmetagraph"]

                samples.append(sample)

                if (idx + 1) % self.batch_size == 0 or (idx + 1) == self.num_iterations:
                    gtg2vid_loss = self.model.gtg2vid_loss(samples)
                    out_log = ""
                    total_loss = 0
                    self.opt.zero_grad()

                    for k, v in gtg2vid_loss.items():
                        out_log += "%s:%.3f\t" % (k, v.item())
                        total_loss += v

                    total_loss.backward()
                    self.opt.step()

                    samples = []
                    global_step += 1

                    if self.use_visual_memory or self.use_semantic_memory:
                        self._flush_memory_logs(mem_log_buffer, global_step)
                        mem_log_buffer = []

                    if (idx + 1) % self.log_freq == 0:
                        print("Epoch: [%d/%d], Iter: [%d/%d]" % (epoch + 1, self.num_epochs, idx + 1, self.num_iterations), out_log)

                    if self.writer is not None:
                        self.writer.add_scalar("Loss/train", total_loss.item(), global_step)

            f1_vid = self.evaluate(global_step)
            if f1_vid >= best_score:
                best_score = f1_vid
                print("Save best weight at epoch: %d" % (epoch + 1))
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                    },
                    os.path.join(self.save_dir, "best_checkpoint.pth"),
                )

            if self.writer is not None:
                self.writer.flush()

    def evaluate(self, global_step=None):
        if not self.is_training:
            checkpoint = torch.load(self.load_dir)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print("Load from epoch %d" % (checkpoint["epoch"]))
            loaders = {"test": self.test_loader}
        else:
            loaders = {"test": self.test_loader}

        self.model.eval()

        with torch.no_grad():
            for loader_name, loader in loaders.items():
                video_pair_list = []
                type_video_pair_list = []
                raw_error_video_pair_list = []
                final_error_video_pair_list = []
                predstep_steps = []
                erm_log_buffer = []

                for video_idx, data in enumerate(loader):
                    v_feature, label, type_label, video = data
                    video = video[0]
                    feature = v_feature.to(self.device)

                    if self.use_visual_memory or self.use_semantic_memory:
                        action_logits, frame_features, aux = self.model.forward_with_aux(
                            feature.permute(0, 2, 1)
                        )
                    else:
                        action_logits, frame_features = self.model(feature.permute(0, 2, 1))
                        aux = None

                    label = label.squeeze(0)
                    type_label = type_label.squeeze(0)
                    error_label = type_label.clone()

                    sample = {}
                    sample["action_logits"] = action_logits.squeeze(0).cpu()
                    sample["label"] = label
                    sample["num_classes"] = self.num_classes
                    sample["video_id"] = video
                    sample["metagraph"] = self.G.graph_info["metagraph"]
                    sample["gmetagraph"] = self.G.graph_info["gmetagraph"]

                    gmetadag = sample["gmetagraph"]
                    sorted_node_ids = list(lexicographical_topological_sort(gmetadag))
                    idx2node = {idx: node_id for idx, node_id in enumerate(sorted_node_ids)}

                    zx_costs, drop_costs, node_drop_costs = compute_generalized_metadag_costs(
                        sample, idx2node, self.drop_base, self.node_drop_base
                    )
                    _, pred, type_pred = generalized_metadag2vid(
                        zx_costs.cpu().numpy(),
                        drop_costs.cpu().numpy(),
                        node_drop_costs.cpu().numpy(),
                        gmetadag,
                        idx2node,
                    )

                    pred[pred >= self.num_classes] = self.bg_idx

                    # second round, no drop
                    zx_costs, drop_costs, node_drop_costs = compute_generalized_metadag_costs(
                        sample, idx2node, -100, -200
                    )
                    _, no_drop_pred_raw, _, no_drop_meta = generalized_metadag2vid(
                        zx_costs.cpu().numpy(),
                        drop_costs.cpu().numpy(),
                        node_drop_costs.cpu().numpy(),
                        gmetadag,
                        idx2node,
                        return_meta_labels=True,
                    )

                    no_drop_pred = np.copy(no_drop_pred_raw)
                    no_drop_pred[no_drop_pred >= self.num_classes] = self.bg_idx

                    if self.use_new_erm:
                        erm_inputs = self._build_new_erm_inputs(
                            pred=pred,
                            no_drop_pred_raw=no_drop_pred_raw,
                            no_drop_meta=no_drop_meta,
                            frame_features=frame_features,
                            aux=aux,
                            video_id=video,
                        )
                        pred, type_pred, error_pred, erm_aux = self.erm_module.forward_with_aux(erm_inputs)

                        raw_error_pred = np.asarray(erm_aux["raw_error_pred_seq"])
                        final_error_pred = np.asarray(erm_aux["final_error_pred_seq"])
                    else:
                        pred, type_pred, error_pred = self.erm(
                            pred,
                            no_drop_pred,
                            feature.permute(0, 2, 1).cpu().squeeze(0),
                        )
                        erm_aux = None
                        raw_error_pred = np.copy(error_pred)
                        final_error_pred = np.copy(error_pred)

                    if self.use_new_erm and erm_aux is not None:
                        erm_log_dict = self._compute_erm_log_dict(erm_aux)
                        if len(erm_log_dict) > 0:
                            erm_log_buffer.append(erm_log_dict)

                        if self.is_vis:
                            self._dump_erm_debug_json(video, erm_aux)

                    label_w_error_cls = label.clone()
                    label_w_error_cls[type_label != 0] = -1

                    vis_dir = "vis"
                    log_dir = "log"

                    if self.is_vis:
                        os.makedirs(os.path.join(self.save_dir, vis_dir), exist_ok=True)
                        os.makedirs(os.path.join(self.save_dir, vis_dir, "gt"), exist_ok=True)

                        draw_pred(label_w_error_cls.long(), "gt", self.draw_idx2action, os.path.join(self.save_dir, vis_dir, "gt", video + "_as"))
                        draw_pred(torch.from_numpy(pred).long(), "as", self.draw_idx2action, os.path.join(self.save_dir, vis_dir, video + "_as"))
                        draw_pred(type_label.squeeze(0).cpu().long(), "gt", self.idx2actiontype, os.path.join(self.save_dir, vis_dir, "gt", video + "_er"))
                        draw_pred(torch.from_numpy(type_pred).long(), "er", self.idx2actiontype, os.path.join(self.save_dir, vis_dir, video + "_er"))
                        draw_pred(error_label.squeeze(0).cpu().long(), "gt", self.idx2actiontype, os.path.join(self.save_dir, vis_dir, "gt", video + "_ed"))
                        draw_pred(torch.from_numpy(final_error_pred).long(), "ed", self.idx2actiontype, os.path.join(self.save_dir, vis_dir, video + "_ed"))
                        draw_pred(torch.from_numpy(raw_error_pred).long(), "ed", self.idx2actiontype, os.path.join(self.save_dir, vis_dir, video + "_ed_raw"))

                        os.makedirs(os.path.join(self.save_dir, "output"), exist_ok=True)
                        os.makedirs(os.path.join(self.save_dir, "output", "tas_nodrop"), exist_ok=True)
                        os.makedirs(os.path.join(self.save_dir, "output", "tas"), exist_ok=True)
                        os.makedirs(os.path.join(self.save_dir, "output", "ed_raw"), exist_ok=True)
                        os.makedirs(os.path.join(self.save_dir, "output", "ed_final"), exist_ok=True)
                        os.makedirs(os.path.join(self.save_dir, "output", "ed"), exist_ok=True)  # legacy alias
                        os.makedirs(os.path.join(self.save_dir, "output", "er"), exist_ok=True)

                        with open(os.path.join(self.save_dir, "output", "tas_nodrop", video + ".txt"), "w") as fp:
                            json.dump(no_drop_pred.tolist(), fp)
                        with open(os.path.join(self.save_dir, "output", "tas", video + ".txt"), "w") as fp:
                            json.dump(pred.tolist(), fp)
                        with open(os.path.join(self.save_dir, "output", "ed_raw", video + ".txt"), "w") as fp:
                            json.dump(raw_error_pred.tolist(), fp)
                        with open(os.path.join(self.save_dir, "output", "ed_final", video + ".txt"), "w") as fp:
                            json.dump(final_error_pred.tolist(), fp)
                        with open(os.path.join(self.save_dir, "output", "ed", video + ".txt"), "w") as fp:
                            json.dump(final_error_pred.tolist(), fp)
                        with open(os.path.join(self.save_dir, "output", "er", video + ".txt"), "w") as fp:
                            json.dump(type_pred.tolist(), fp)

                    video_pair_list.append(Video(video_idx, pred.tolist(), label_w_error_cls.tolist()))
                    type_video_pair_list.append(Video(video_idx, type_pred.tolist(), type_label.cpu().tolist()))

                    error_label[error_label > 0] = 1
                    raw_error_video_pair_list.append(Video(video_idx, raw_error_pred.tolist(), error_label.tolist()))
                    final_error_video_pair_list.append(Video(video_idx, final_error_pred.tolist(), error_label.tolist()))

                    pred_for_omit = pred.copy()
                    pred_for_omit[pred_for_omit == -1] = self.bg_idx
                    label_for_omit = label.clone()
                    label_for_omit[label_for_omit == -1] = self.bg_idx
                    steps, _ = self.from_framewise_to_steps(label_for_omit, ignore_bg=True)
                    pred_steps, _ = self.from_framewise_to_steps(torch.tensor(pred_for_omit).long(), ignore_bg=True)
                    predstep_steps.append([pred_steps, steps])

                if self.naming == "EgoPER":
                    oIoU, oAcc = omission_detection(self.G.graph_info["graph"], predstep_steps)
                    omit_log = []
                    omit_log.append("Omission Detecion:\n")
                    omit_log.append("|oIoU:%.1f|oAcc:%.1f|" % (oIoU * 100, oAcc * 100))
                else:
                    omit_log = None

                # AS
                ckpt = Checkpoint(bg_class=[self.ignore_idx])
                ckpt.add_videos(video_pair_list)
                as_out, as_per_out = ckpt.compute_metrics()
                as_out_log = "|Edit:%.1f|Acc:%.1f|" % (as_out["edit"] * 100, as_out["acc"] * 100)
                as_per_out_log, as_avg_f1 = self.compute_per_out_log(as_per_out, mode="as")

                # ER
                ckpt = Checkpoint(bg_class=[self.ignore_idx])
                ckpt.add_videos(type_video_pair_list)
                er_out, er_per_out = ckpt.compute_metrics()
                er_out_log = "|Error Recognition|F1@.1:%.1f|F1@.25:%.1f|F1@.5:%.1f|" % (
                    er_out["F1@0.100"] * 100,
                    er_out["F1@0.250"] * 100,
                    er_out["F1@0.500"] * 100,
                )
                er_per_out_log, er_avg_f1 = self.compute_per_out_log(er_per_out, use_ignore=True, mode="er")

                # raw ED
                ckpt = Checkpoint(bg_class=[self.ignore_idx])
                ckpt.add_videos(raw_error_video_pair_list)
                raw_ed_out, raw_ed_per_out = ckpt.compute_metrics()
                raw_ed_out_log = "|Raw Error Detection|F1@.1:%.1f|F1@.25:%.1f|F1@.5:%.1f|" % (
                    raw_ed_out["F1@0.100"] * 100,
                    raw_ed_out["F1@0.250"] * 100,
                    raw_ed_out["F1@0.500"] * 100,
                )
                raw_ed_per_out_log, raw_ed_avg_f1 = self.compute_per_out_log(raw_ed_per_out, mode="ed")

                # final ED
                ckpt = Checkpoint(bg_class=[self.ignore_idx])
                ckpt.add_videos(final_error_video_pair_list)
                final_ed_out, final_ed_per_out = ckpt.compute_metrics()
                final_ed_out_log = "|Final Error Detection|F1@.1:%.1f|F1@.25:%.1f|F1@.5:%.1f|" % (
                    final_ed_out["F1@0.100"] * 100,
                    final_ed_out["F1@0.250"] * 100,
                    final_ed_out["F1@0.500"] * 100,
                )
                final_ed_per_out_log, final_ed_avg_f1 = self.compute_per_out_log(final_ed_per_out, mode="ed")

                os.makedirs(os.path.join(self.save_dir, log_dir), exist_ok=True)

                as_out_logs = list(as_per_out_log)
                as_out_logs.append("\n\n")
                as_out_logs.append(as_out_log)

                er_out_logs = list(er_per_out_log)
                er_out_logs.append("\n\n")
                er_out_logs.append(er_out_log)

                raw_ed_out_logs = list(raw_ed_per_out_log)
                raw_ed_out_logs.append("\n\n")
                raw_ed_out_logs.append(raw_ed_out_log)
                raw_ed_out_logs.append("\n\n")
                if omit_log is not None:
                    raw_ed_out_logs.extend(omit_log)

                final_ed_out_logs = list(final_ed_per_out_log)
                final_ed_out_logs.append("\n\n")
                final_ed_out_logs.append(final_ed_out_log)
                final_ed_out_logs.append("\n\n")
                if omit_log is not None:
                    final_ed_out_logs.extend(omit_log)

                with open(os.path.join(self.save_dir, log_dir, "action_segmentation.txt"), "w") as fp:
                    fp.writelines(as_out_logs)
                with open(os.path.join(self.save_dir, log_dir, "error_recognition.txt"), "w") as fp:
                    fp.writelines(er_out_logs)
                with open(os.path.join(self.save_dir, log_dir, "error_detection_raw.txt"), "w") as fp:
                    fp.writelines(raw_ed_out_logs)
                with open(os.path.join(self.save_dir, log_dir, "error_detection_final.txt"), "w") as fp:
                    fp.writelines(final_ed_out_logs)
                with open(os.path.join(self.save_dir, log_dir, "error_detection.txt"), "w") as fp:
                    fp.writelines(final_ed_out_logs)

                if self.writer is not None:
                    self.writer.add_scalar("AS_F1@0.500/valid", as_out["F1@0.500"] * 100, global_step)
                    self.writer.add_scalar("ER_F1@0.500/valid", er_out["F1@0.500"] * 100, global_step)
                    self.writer.add_scalar("ED_RAW_F1@0.500/valid", raw_ed_out["F1@0.500"] * 100, global_step)
                    self.writer.add_scalar("ED_FINAL_F1@0.500/valid", final_ed_out["F1@0.500"] * 100, global_step)
                    self.writer.add_scalar("AVG_ER_F1/valid", er_avg_f1, global_step)
                    self.writer.add_scalar("AVG_ED_RAW_F1/valid", raw_ed_avg_f1, global_step)
                    self.writer.add_scalar("AVG_ED_FINAL_F1/valid", final_ed_avg_f1, global_step)

                    if self.use_new_erm and len(erm_log_buffer) > 0:
                        keys = erm_log_buffer[0].keys()
                        for k in keys:
                            mean_v = float(np.mean([item[k] for item in erm_log_buffer]))
                            self.writer.add_scalar(k, mean_v, global_step)

                    if self.is_vis:
                        grid = create_image_grid(os.path.join(self.save_dir, vis_dir))
                        self.writer.add_image("Images/valid", grid, global_step)

            self.model.train()
            print("Evalutation Done...")
            return (as_avg_f1 + final_ed_avg_f1 + er_avg_f1) / 3