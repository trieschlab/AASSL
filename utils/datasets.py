import csv
import io
import math
import os
import time
import random

import scipy
import h5py
import pandas as pd
import numpy as np
import torch
from torch.linalg import lstsq

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


from utils.general import str2bool, str2table, get_representations
from utils.logger import EpochLogger




class SimpleDataset(Dataset):

    def __init__(self, args, run_name, split='train', transform=None, target_transform=None, contrastive=True, logger=True, fabric=None, eval=False, **kwargs):
        self.args = args
        self.contrastive = contrastive
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.fabric=fabric
        self.run_name = run_name
        self.eval_mode = eval
        if split == "test" and logger and fabric.global_rank == 0:
            self.epoch_logger = EpochLogger(output_dir=os.path.join(args.log_dir, run_name),
                                            exp_name="Seed-" + str(args.seed), output_fname='progress.txt')
            self.epoch_logger.save_config(args)
        if self.args.unijit:
            s = self.args.jitter_strength
            jit = transforms.RandomApply([transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)], p=self.args.jitter)
            grayscale = transforms.RandomGrayscale(p=0.2)
            self.jit = transforms.Compose([jit, grayscale])

        self.time = time.time()

    @classmethod
    def get_args(cls, parser):
        return parser

    @torch.no_grad()
    def eval(self, net, dataloader_train_eval, dataloader_test, epoch=0, modules=[], tv=None, **kwargs):
        if self.fabric.global_rank == 0:
            test_time = time.time()

        data_test = get_representations(self.args, net, dataloader_test, tv)
        data_test = self.fabric.all_gather(data_test)

        if self.fabric.global_rank == 0:
            self.epoch_logger.log_tabular("epoch", epoch)
            self.epoch_logger.log_tabular("all_time", (time.time() - self.time) / self.args.test_every)
            self.epoch_logger.log_tabular("test_time", (time.time() - test_time) / self.args.test_every)

            f_l_test = {}
            if "0" in self.args.eval_labels:
                f_l_test["0"] = {"features": data_test[0], "labels": data_test[1], "supervised": data_test[-1]}

            for m in modules:
                for k, v in m.eval(net, f_l_test).items():
                    self.epoch_logger.log_tabular(k, v)
            self.time = time.time()
            self.epoch_logger.dump_tabular()
        self.fabric.barrier()

    def get_log_dir(self):
        return os.path.join(self.args.log_dir, self.run_name)

class Toys4kDataset(SimpleDataset):
    def __init__(self, *args, view=None, category=None, object=None, **kwargs):
        super().__init__(*args,  **kwargs)
        self.hdf5_file = h5py.File(os.path.join(self.args.data_root, "data.h5"), 'r')
        annotations_file = "dataset.parquet" if self.split == "train" else "dataset_test.parquet"
        self.img_labels = pd.read_parquet(os.path.join(self.args.data_root, annotations_file))


        self.num_views = 180
        self.max_views = 180
        self.view = None
        self.img_labels.columns = self.img_labels.columns.astype(str)

        if self.args.finetune_labels != -1 and self.split == "train":
            self.img_labels = self.img_labels.groupby("2").apply(lambda x: x.sample(self.args.finetune_labels)).reset_index(drop=True)

        if self.eval_mode:
            self.img_labels = self.img_labels.groupby("2").apply(lambda x: x.sample(1)).reset_index(drop=True)


        result = [int(r.split("_")[-3]) for r in self.img_labels["0"]]
        self.img_labels["12"] = result
        if view is not None:
            self.img_labels = self.img_labels.loc[self.img_labels.iloc[:, 12] == view]
            self.img_labels.reset_index(inplace=True)

        if category is not None:
            self.img_labels = self.img_labels.loc[self.img_labels.loc[:, "5"] == category]
            self.img_labels.reset_index(inplace=True)

        elif object is not None:
            self.img_labels = self.img_labels.loc[self.img_labels.loc[:, "2"] == object]
            self.img_labels.reset_index(inplace=True)


        self.replace = "binoc" in self.args.data_root
        self.num_columns = len(self.img_labels.columns)
        self.log_backgrounds = not ("back5_" in self.args.data_root or "back7_" in self.args.data_root)


        i, j, k = 0, 0, 0
        self.obj_to_int = {}
        self.cat_to_int = {}
        self.category_list = []
        self.object_list = []
        for index, _ in self.img_labels.groupby("2").count().iterrows():
            self.obj_to_int[index] = i
            self.object_list.append(index)
            i += 1
        for index, _ in self.img_labels.groupby("5").count().iterrows():
            self.cat_to_int[index] = j
            self.category_list.append(index)
            j += 1
        self.back_to_int = {}
        for index, _ in self.img_labels.groupby("3").count().iterrows():
            self.back_to_int[index] = k
            k += 1
        self.max_backgrounds = len(self.back_to_int)
        self.n_classes = len(self.cat_to_int)
        self.n_objs = len(self.obj_to_int)

    @classmethod
    def get_args(cls, parser):
        return parser

    def __len__(self):
        return len(self.img_labels)

    def get_actions(self, idx, r, idx2):
        if self.args.sampling_mode == "randomwalk":
            return torch.tensor([r, r], dtype=torch.float32)


        true_rotation = r * 360 / self.max_views
        rad_rotation = math.pi * true_rotation / 180
        return torch.tensor([math.sin(rad_rotation), math.cos(rad_rotation)])



    def get_image(self, idx):
        if self.args.hdf5:
            h5_index = self.img_labels.loc[idx, "h5_index"]
            return Image.open(io.BytesIO(self.hdf5_file[self.split][h5_index])),

        img_path = os.path.join(self.args.data_root, self.img_labels.loc[idx, "2"], self.img_labels.loc[idx, "0"])
        if self.replace:
            img_path = img_path.replace(".png", "_vc.png")
        return Image.open(img_path),

    def get_other_image(self, idx):
        splitted = self.img_labels.loc[idx, "0"].split("_")
        original_pos = int(splitted[-3])
        if self.args.sampling_mode == "randomwalk+":
            rotation = 1
        elif self.args.sampling_mode == "randomwalk":
            rotation = (1 if random.random() < 0.5 else -1)
        elif self.args.sampling_mode == "opposite":
            rotation = self.num_views/2
        elif self.args.sampling_mode == "uniform":
            rotation = random.randint(-self.num_views, self.num_views)
        nv = original_pos + rotation
        if nv > self.max_views:
            pos = nv - self.max_views
        elif nv <= 0:
            pos = nv + self.max_views
        else:
            pos = nv

        new_idx = idx + pos - original_pos
        action = self.get_actions(idx, rotation, idx + pos - original_pos)
        return *self.get_image(new_idx), action


    def __getitem__(self, idx):
        image, = self.get_image(idx)
        label = self.cat_to_int[self.img_labels.loc[idx, "5"]]
        label_back = self.back_to_int[self.img_labels.loc[idx, "3"]]
        label_obj = self.obj_to_int[self.img_labels.loc[idx, "2"]]
        # label_view = int(self.img_labels.iloc[idx, 0].split("_")[-3])
        label_view = int(self.img_labels.loc[idx, "12"])
        if self.target_transform:
            label = self.target_transform(label)
        state = torch.get_rng_state()
        if self.args.unijit and self.contrastive:
            image = self.transform(image)
            img_pair, a = self.get_other_image(idx)
            img_pair = self.transform(img_pair)
            state = torch.get_rng_state()
            image_t = self.jit(image)
            if random.random() <= self.args.punijit:
                torch.set_rng_state(state)
            img_pair_t = self.jit(img_pair)


            return (image_t, img_pair_t, a), label,label_back, label_obj

        if self.args.unijit:
            image_t = self.transform(image)
            img_pair_t = self.transform(image)
            state = torch.get_rng_state()
            image_t = self.jit(image_t)
            torch.set_rng_state(state)
            img_pair_t = self.jit(img_pair_t)
            return (image_t, img_pair_t, torch.zeros(2, )), label, label_back, label_obj, label_view

        if self.transform:
            image_t = self.transform(image)

        if self.transform and self.contrastive:
            img_pair, a = self.get_other_image(idx)#if random.random() < self.args.p_time else (image, torch.zeros((2,)))
            img_pair_t = self.transform(img_pair)

        if not self.contrastive:
            return (image_t, image_t, torch.zeros(2,)), label, label_back, label_obj, label_view


        return (image_t, img_pair_t, a), label, label_back, label_obj, label_view


    @torch.no_grad()
    def eval(self, net, dataloader_train_eval, dataloader_test, epoch=0, scheduler=None, modules=[], dataset_train=None, tv=None, **kwargs):
        test_time = time.time()

        #Compute features
        # data = get_representations(self.args, net, dataloader_train_eval)
        # if "2" in self.args.eval_labels:
        #     f_obj_train, f_obj_test = data[0][dataset_train.obj_split_train], data[0][dataset_train.obj_split_test]
        #     labels_obj_train, labels_obj_test = data[3][dataset_train.obj_split_train], data[3][dataset_train.obj_split_test]
        #     f_l_test["2"] = {"features": f_obj_test, "labels": labels_obj_test}

        data = get_representations(self.args, net, dataloader_train_eval, tv)
        features_train_eval, labels_train_eval = data[0], data[1]
        lstsq_model = lstsq(features_train_eval, torch.nn.functional.one_hot(labels_train_eval, self.n_classes).type(torch.float32))


        data_test = get_representations(self.args, net, dataloader_test, tv)
        f_l_test={}
        if "0" in self.args.eval_labels:
            f_l_test["0"] = {"features": data_test[0], "labels": data_test[1], "supervised": data_test[-1]}

            mask_side = data_test[4] == self.max_views
            f_l_test["_side"] = {"features": data_test[0][mask_side], "labels": data_test[1][mask_side]}

            mask_face = data_test[4] == 135
            f_l_test["_face"] = {"features": data_test[0][mask_face], "labels": data_test[1][mask_face]}

            mask_quarter = data_test[4] == 157
            f_l_test["_quarter"] = {"features": data_test[0][mask_quarter], "labels": data_test[1][mask_quarter]}

        pred = data_test[0] @ lstsq_model.solution
        mean_acc = (pred.argmax(-1) == data_test[1]).to(torch.float).mean()

        self.epoch_logger.log_tabular("test_acc1", mean_acc.item())
        self.epoch_logger.log_tabular("sparse", torch.count_nonzero(data_test[0], dim=1).to(torch.float32).mean(dim=0).item())
        self.epoch_logger.log_tabular("epoch", epoch)
        self.epoch_logger.log_tabular("all_time", (time.time()-self.time)/self.args.test_every)
        self.epoch_logger.log_tabular("test_time", (time.time()-test_time)/self.args.test_every)
        for m in modules:
            for k, v in m.eval(net, f_l_test).items():
                self.epoch_logger.log_tabular(k, v)


        self.time = time.time()
        self.epoch_logger.dump_tabular()

class CO3D(SimpleDataset):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from utils.data_types import CO3D_CATEGORIES
        assert self.args.hdf5, "only hdf5 available"
        self.hdf5_file = h5py.File(os.path.join(self.args.data_root, "data.h5"), 'r')
        self.dataset = pd.read_parquet(os.path.join(self.args.data_root, "dataset.parquet" if self.split!="test" else "dataset_test.parquet"))
        self.t = 0
        self.cat_to_int = {}
        self.category_list = []
        j=0

        if self.args.finetune_labels != -1 and self.split == "train":
            self.dataset = self.dataset.groupby("object").apply(lambda x: x.sample(self.args.finetune_labels)).reset_index(drop=True)

        if self.split == "train":
            dataset2 = pd.read_parquet(os.path.join(self.args.data_root,"dataset_test.parquet"))
            dd2 = dataset2.groupby("object").first().to_dict()
            dd1 = self.dataset.groupby("object").first().to_dict()
            for k, r in dd2["path"].items():
                assert k not in dd1["path"], "Test and train dataset overlap"

        for index, _ in self.dataset.groupby("category").count().iterrows():
            self.cat_to_int[index] = j
            self.category_list.append(index)

            j += 1
        # self.n_classes = len(self.category_list) if self.args.mode != "finetune" else 51
        self.n_classes = 51
        self.action_headers = np.array(["a"+str(1+i) for i in range(14)])
        self.action_quatheaders = np.array(["a"+str(1+i) for i in range(9)])
        self.action_foclength_trans = np.array(["a"+str(1+i) for i in range(9,14)])
        if self.args.co3d_quaternion:
            self.action_mean = torch.tensor([0.0057,-0.0274,0.168, 0.545]+[-1.16e+06, 2.71e+05, 4.83e+05,-0.000415,-0.000257])
            self.action_std = torch.tensor([0.0841, 0.404, 0.499, 0.505]+[ 3.38e+09, 7.88e+08,1.4e+09, 9.26, 4.94])
        else:
            self.action_std = torch.tensor([9.66991054e-01, 7.18770433e-01, 5.58006881e-01, 7.14118006e-01,6.62875426e-01, 4.25014783e-01, 5.67008818e-01, 4.29122972e-01,4.37485732e-01, 3.02687459e+09, 7.04659724e+08, 1.25568361e+09,1.03948843e+01, 5.53417153e+00])
            self.action_mean = torch.tensor([-0.00023,-0.0001,0.00003, 0.00004,0.00010,-0.00037,0.00001, 0.00026, 0.00018, 2326900, -541714, -965291, -0.00240, -0.00142])
        print(len(self.dataset), self.split, self.n_classes)



    def __len__(self):
        return len(self.dataset)

    @classmethod
    def get_args(cls, parser):
        parser.add_argument("--co3d_normalize", type=str2bool,default=True)
        parser.add_argument("--co3d_quaternion", type=str2bool,default=True)
        return parser


    def open_image(self, category, obj, index, get_size=False):
        if get_size:
            npar = self.hdf5_file.get(category).get(obj)
            return Image.open(io.BytesIO(npar[index])), len(npar)
        return Image.open(io.BytesIO(self.hdf5_file.get(category).get(obj)[index]))


    def get_action(self, idx, new_idx):
        if self.args.co3d_quaternion:
            q1 = torch.tensor(self.dataset.loc[idx, self.action_quatheaders].values.astype(np.float32)).view(3,3)
            q2 = torch.tensor(self.dataset.loc[new_idx, self.action_quatheaders].values.astype(np.float32)).view(3,3)
            qtrans = torch.matmul(torch.transpose(q1, 1, 0), q2)
            r = scipy.spatial.transform.Rotation.from_matrix(qtrans)
            qtrans = torch.tensor(r.as_quat(), dtype=torch.float32)

            diff_translength = torch.tensor(self.dataset.loc[new_idx, self.action_foclength_trans].values.astype(np.float32) - self.dataset.loc[idx, self.action_foclength_trans].values.astype(np.float32))
            action = torch.cat((qtrans, diff_translength), dim=0)
            return action

        a0 = torch.tensor(self.dataset.loc[idx, self.action_headers].values.astype(np.float32),dtype=torch.float32)
        a1 = torch.tensor(self.dataset.loc[new_idx, self.action_headers].values.astype(np.float32),dtype=torch.float32)
        action = a0 - a1
        return action

    def __getitem__(self, idx):
        category, obj, frame_index = self.dataset.loc[idx, "category"],self.dataset.loc[idx, "object"],self.dataset.loc[idx, "index"]

        image, size = self.open_image(category, obj, frame_index, get_size=True)
        label = self.cat_to_int[category]
        if self.transform:
            image_t = self.transform(image)

        if not self.contrastive:
            return (image_t, self.transform(image) if self.split == "train" else image_t, torch.zeros((14,))), label

        if self.args.sampling_mode == "uniform":
            new_frame_index = random.randint(0, size-1)
        elif self.args.sampling_mode == "randomwalk+":
            new_frame_index = frame_index+1 if frame_index < size-1 else frame_index - 1
        else:
            new_frame_index = max(0, min(size-1, (frame_index-1 if random.random() < 0.5 else frame_index+1)))

        new_idx = idx + new_frame_index - frame_index
        assert str(obj) == str(self.dataset.loc[new_idx, "object"])
        image_pair = self.open_image(category, obj, self.dataset.loc[new_idx, "index"])

        action = self.get_action(idx, new_idx)
        if self.args.co3d_normalize:
            action = (action - self.action_mean)/self.action_std

        if self.transform:
            image_pair = self.transform(image_pair)
        return (image_t, image_pair, action), label


class MVImgNet(SimpleDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.category_mapping = {}
        category_list = []
        for index, v in pd.read_csv("utils/mvimgnet_category.txt", header=None).iterrows():
            self.category_mapping[v[0]] = index
            category_list.append(v[0])
        self.n_classes = len(self.category_mapping)

        if self.args.hdf5 and self.args.hdf5_mode == "partition":
            if not os.path.exists(os.path.join(self.args.data_root, "data_all.h5")):
                raise Exception("You should build the merged data hdf5 file")
            self.hdf5_file = h5py.File(os.path.join(self.args.data_root, "data_all.h5"), "r")
            self.dataset = pd.read_parquet(os.path.join(self.args.data_root, f"dataset_{self.split}_all3.parquet"))
        elif self.args.hdf5:
            self.hdf5_file = h5py.File(os.path.join(self.args.data_root, "data2.h5"), "r")
            self.dataset = pd.read_parquet(os.path.join(self.args.data_root, "dataset_" + self.split + "2.parquet"))
        else:
            self.dataset = pd.read_parquet(os.path.join(self.args.data_root, "dataset_" + self.split + "2.parquet"))



        if self.args.imgnet_subset and self.split == "train":
            uniqs = self.dataset["object"].unique()
            pathobj = os.path.join(self.args.data_root, f"subset_{self.args.seed}_{self.args.imgnet_subset}")
            if self.fabric.global_rank == 0:
                if not os.path.exists(pathobj):
                    sampled_objects = np.random.choice(uniqs, int(len(uniqs)*self.args.imgnet_subset))
                    f = open(pathobj, "w")
                    csv.writer(f).writerow(sampled_objects)
                    f.close()

            self.fabric.barrier()
            f = open(pathobj, "r")
            sampled_objects = next(iter(csv.reader(f)))
            f.close()
            self.dataset = self.dataset.query('object in @sampled_objects').reset_index(drop=True)

        if self.args.finetune_labels != -1 and self.split == "train":
            self.dataset = self.dataset.groupby("object").apply(lambda x: x.sample(self.args.finetune_labels)).reset_index(drop=True)

        self.action_headers = np.array(["q0","q1","q2","q3","t0","t1","t2"])
        self.action_mean = torch.tensor([0.759, -0.000354, -0.00682, -0.00723 , 0.00314, 0.00787, -0.0171, 0])
        self.action_std = torch.tensor([0.431, 0.048, 0.358, 0.328, 3.8, 1.25, 1.13, 1])

        print(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def open_image(self, category, obj, index, path, idx):
        if self.args.hdf5_mode == "partition":
            partition = self.dataset.loc[idx, "partition"]
            p = self.hdf5_file.get(partition)
            c = p.get(category)
            o = c.get(obj)
            return Image.open(io.BytesIO(o[index]))


        c = self.hdf5_file.get(category)
        o = c.get(obj)
        return Image.open(io.BytesIO(o[index]))


    @classmethod
    def get_args(cls, parser):
        parser.add_argument("--hdf5_mode", type=str, default="normal")
        parser.add_argument("--action_clamp", type=str2bool, default=True)
        parser.add_argument("--imgnet_subset", type=float, default=0)
        return parser

    def quaternion_multiply(self, quaternion1, quaternion0):
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

    def get_action(self, idx, new_idx, image):
        a0 = self.dataset.loc[idx, self.action_headers].values.astype(np.float32)
        a1 = self.dataset.loc[new_idx, self.action_headers].values.astype(np.float32)

        q0, q1 = np.concatenate((a0[1:4],a0[0:1]), axis=0), np.concatenate((a1[1:4],a1[0:1]), axis=0)
        resq = self.quaternion_multiply((q0[3:4],-q0[0:1],-q0[1:2],-q0[2:3]), (q1[3:4],q1[0:1],q1[1:2],q1[2:3])).squeeze()
        imsizeM = np.array([float((image.size[0] > image.size[1]))])
        if not self.args.action_clamp:
            action = torch.tensor(np.concatenate((resq, a1[4:] - a0[4:], imsizeM), axis=0), dtype=torch.float32)
        else:
            action = torch.tensor(np.concatenate((resq, np.clip(a1[4:] - a0[4:], -50, 50), imsizeM), axis=0), dtype=torch.float32)

        action = (action - self.action_mean)/self.action_std
        return action

    def __getitem__(self, idx):
        # idx=530
        category, obj, frame_index = self.dataset.loc[idx, "category"],self.dataset.loc[idx, "object"],self.dataset.loc[idx, "frame"]
        size = self.dataset.loc[idx, "length"]
        image = self.open_image(str(category), str(obj),frame_index, self.dataset.loc[idx, "path"], idx)

        label = np.int64(self.category_mapping[category])

        if self.transform:
            image_t = self.transform(image)

        ft = self.args.mode in ["finetune","finetune_all"]
        if not self.contrastive or ft:
            return (image_t, self.transform(image) if self.split == "train" and not ft else image_t, torch.zeros((8,))), label

        if self.args.sampling_mode == "uniform":
            new_frame_index = random.randint(0, size-1)
        elif self.args.sampling_mode == "randomwalk+":
            new_frame_index = frame_index+1 if frame_index < size-1 else frame_index - 1
        else:
            new_frame_index = max(0, min(size-1, (frame_index-1 if random.random() < 0.5 else frame_index+1)))

        new_idx = idx + new_frame_index - frame_index

        assert str(obj) == str(self.dataset.loc[new_idx, "object"]), f"{str(obj)}, {str(self.dataset.loc[new_idx, 'object'])}, {idx}, {new_idx}, {frame_index}, {new_frame_index}, {size}"
        image_pair = self.open_image(str(category), str(obj), self.dataset.loc[new_idx, "frame"],self.dataset.loc[new_idx, "path"], new_idx)

        action = self.get_action(idx, new_idx, image)

        if self.transform:
            image_pair = self.transform(image_pair)
        return (image_t, image_pair, action), label

