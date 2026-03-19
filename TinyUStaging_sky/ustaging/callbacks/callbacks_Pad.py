import numpy as np
import pandas as pd
from carbontracker.tracker import CarbonTracker
from mpunet.utils.plotting import imshow_with_label_overlay
from tensorflow.keras.callbacks import Callback
from ustaging.utils import get_memory_usage
from mpunet.utils import highlighted
from mpunet.logging import ScreenLogger
from collections import defaultdict
from datetime import timedelta

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import psutil
import os
from tensorflow.keras.callbacks import Callback
from datetime import datetime
from mpunet.logging import ScreenLogger
from mpunet.utils.plotting import (imshow_with_label_overlay, imshow,
                                   plot_all_training_curves)



class Validation(Callback):
    """
    Validation computation callback.
    Samples a number of validation batches from a deepsleep
    ValidationMultiSequence object
    and computes for all tasks:
        - Batch-wise validation loss
        - Batch-wise metrics as specified in model.metrics_tensors
        - Epoch-wise pr-class and average precision
        - Epoch-wise pr-class and average recall
        - Epoch-wise pr-class and average dice coefficients
    ... and adds all results to the log dict
    Note: The purpose of this callback over the default tf.keras evaluation
    mechanism is to calculate certain metrics over the entire epoch of data as
    opposed to averaged batch-wise computations.
    """
    def __init__(self,
                 val_sequence,
                 max_val_studies_per_dataset=20,
                 logger=None, verbose=True):
        """
        Args:
            val_sequence: A deepsleep ValidationMultiSequence object
            logger:       An instance of a MultiPlanar Logger that prints to
                          screen and/or file
            verbose:      Print progress to screen - OBS does not use Logger
        """
        super().__init__()
        self.logger = logger or ScreenLogger()
        self.sequences = val_sequence.sequences
        self.verbose = verbose
        self.max_studies = max_val_studies_per_dataset
        self.n_classes = val_sequence.n_classes
        self.IDs = val_sequence.IDs
        self.print_round = 3
        self.log_round = 4
        self._supports_tf_logs = True

    def _compute_counts(self, pred, true, ignore_class=None):
        # Argmax and CM elements
        pred = pred.argmax(-1).ravel()
        true = true.ravel()

        if ignore_class:
            mask = np.where(true != ignore_class)
            true = true[mask]
            pred = pred[mask]

        # ljy注释：【bincount】
        # https://blog.csdn.net/xlinsist/article/details/51346523

        # Compute relevant CM elements
        # We select the number following the largest class integer when
        # y != pred, then bincount and remove the added dummy class
        tps = np.bincount(np.where(true == pred, true, self.n_classes),
                          minlength=self.n_classes+1)[:-1].astype(np.uint64)
        rel = np.bincount(true, minlength=self.n_classes).astype(np.uint64)
        sel = np.bincount(pred, minlength=self.n_classes).astype(np.uint64)
        return tps, rel, sel

    def predict(self):
        # Get tensors to run and their names
        # metrics = self.model.loss_functions + self.model.metrics  # ljy: 出错 没有loss_functions
        # metrics = self.model.metrics  # + self.model.loss # ljy注释：https://github.com/perslev/U-Time/blob/utime-latest/utime/callbacks/callbacks.py
        metrics = getattr(self.model, "loss_functions", self.model.losses) + self.model.metrics
        # ↑ ljy改1128：self.model.losses【trainer.compile_model中√】?  self.model.loss【yaml中×】
        metrics_names = self.model.metrics_names
        self.model.reset_metrics()
        #❤CBAM中的kernel_regularizer=regularizers.l2(0.001)不可加？？？？？？
        # self.logger(f"len(metrics_names):{len(metrics_names)} == len(metrics){len(metrics)}: {len(metrics_names) == len(metrics)}")  # ljy-DEBUG：AssertionError
        assert len(metrics_names) == len(metrics)


        ############# ljy DEBUG 20211013-1：
        # self.logger("\n---------------- self.model.metrics：" , self.model.metrics)
        # self.logger("\n---------------- getattr(losses)：" , getattr(self.model, "loss_functions", self.model.losses))
        # self.logger("\n---------------- metrics_names：" , metrics_names) # 如: val_loss, val_dice, val_recall
        ################################

        # Prepare arrays for CM summary stats
        true_pos, relevant, selected, metrics_results = {}, {}, {}, {}
        for id_, sequence in zip(self.IDs, self.sequences):
            # Add count arrays to the result dictionaries
            true_pos[id_] = np.zeros(shape=(self.n_classes,), dtype=np.uint64)
            relevant[id_] = np.zeros(shape=(self.n_classes,), dtype=np.uint64)
            selected[id_] = np.zeros(shape=(self.n_classes,), dtype=np.uint64)

            # Get validation sleep study loader
            n_val = min(len(sequence.dataset_queue), self.max_studies)
            study_iterator = sequence.dataset_queue.get_study_iterator(n_val)

            # Predict and evaluate on all studies
            per_study_metrics = defaultdict(list)
            for i, sleep_study_context in enumerate(study_iterator):
                if self.verbose:
                    s = "   {}Validation subject: {}/{}".format(f"[{id_}] "
                                                                if id_ else "",
                                                                i+1,
                                                                n_val)
                    print(s, end="\r", flush=True)

                with sleep_study_context as ss:
                    x, y = sequence.get_single_study_full_seq(ss.identifier,
                                                              reshape=True)
                    pred = self.model.predict_on_batch(x)

                # Compute counts （ljy：转到numpy上）
                if hasattr(pred, "numpy"):
                    pred_numpy = pred.numpy()
                else:
                    pred_numpy = pred

                # #### TODO: ljy加 -- 绘制置信度： #########
                # self.logger("\n================== ljy：plot_confidence")
                # self.logger("pred.shape:", pred.shape)
                # self.logger("pred:", pred)
                #
                # self.logger("\n\n\ntrue.shape:", true.shape)
                # self.logger("true:", true)
                #
                #
                # #################

                tps, rel, sel = self._compute_counts(pred=pred_numpy,
                                                     true=y,
                                                     ignore_class=5)
                true_pos[id_] += tps
                relevant[id_] += rel
                selected[id_] += sel

                # Run all metrics
                for metric, name in zip(metrics, metrics_names):
                    res = metric(y, pred)
                    if hasattr(pred, "numpy"):
                        res = res.numpy()
                    per_study_metrics[name].append(res)

                    ############# ljy DEBUG 20211013-1：
                    # self.logger(f"\n--------- ljy:  metric: name = {metric} : {name}")
                    # self.logger(f"\n---------------- metric(y, pred).numpy()：{res}")
                    # self.logger(f"\n----------------  ↑ {metric(y, pred)}") # 如: loss, sparse_categorical_accuracy
                    ################################

            # Compute mean metrics for the dataset
            metrics_results[id_] = {}
            for metric, name in zip(metrics, metrics_names):
                metrics_results[id_][name] = np.mean(per_study_metrics[name])
            self.model.reset_metrics()
            self.logger("")
        self.logger("")
        return true_pos, relevant, selected, metrics_results

    @staticmethod
    def _compute_dice(tp, rel, sel):
        # Get data masks (to avoid div. by zero warnings)
        # We set precision, recall, dice to 0 in for those particular cls.
        sel_mask = sel > 0
        rel_mask = rel > 0

        # prepare arrays
        precisions = np.zeros(shape=tp.shape, dtype=np.float32)
        recalls = np.zeros_like(precisions)
        dices = np.zeros_like(precisions)

        # Compute precisions, recalls
        precisions[sel_mask] = tp[sel_mask] / sel[sel_mask]
        recalls[rel_mask] = tp[rel_mask] / rel[rel_mask]

        # Compute dice
        intrs = (2 * precisions * recalls)
        union = (precisions + recalls)
        dice_mask = union > 0
        dices[dice_mask] = intrs[dice_mask] / union[dice_mask]

        return precisions, recalls, dices

    def _print_val_results(self, precisions, recalls, dices, metrics, epoch,
                           name, classes):
        # Log the results
        # We add them to a pd dataframe just for the pretty print output
        index = ["cls %i" % i for i in classes]
        metric_keys, metric_vals = map(list, list(zip(*metrics.items())))
        col_order = metric_keys + ["precision", "recall", "dice"]
        nan_arr = np.empty(shape=len(precisions))
        nan_arr[:] = np.nan
        value_dict = {"precision": precisions,
                      "recall": recalls,
                      "dice": dices}
        value_dict.update({key: nan_arr for key in metrics})
        val_results = pd.DataFrame(value_dict,
                                   index=index).loc[:, col_order]  # ensure order
        # Transpose the results to have metrics in rows
        val_results = val_results.T
        # Add mean and set in first row
        means = metric_vals + [precisions.mean(), recalls.mean(), dices.mean()]
        val_results["mean"] = means
        cols = list(val_results.columns)
        cols.insert(0, cols.pop(cols.index('mean')))
        val_results = val_results.loc[:, cols]

        # Print the df to screen
        self.logger(highlighted(("[%s] Validation Results for "
                                 "Epoch %i" % (name, epoch)).lstrip(" ")))
        print_string = val_results.round(self.print_round).to_string()
        self.logger(print_string.replace("NaN", "---") + "\n")

    # ljy： 计算这一个batch的mean_metrics
    def on_epoch_end(self, epoch, logs=None):
        self.logger("\n")
        # Predict and get CM
        TPs, relevant, selected, metrics = self.predict()
        for id_ in self.IDs:
            tp, rel, sel = TPs[id_], relevant[id_], selected[id_]
            precisions, recalls, dices = self._compute_dice(tp=tp, sel=sel, rel=rel)
            classes = np.arange(len(dices))

            # Add to log
            n = (id_ + "_") if len(self.IDs) > 1 else ""  # ljy注释： id--数据库, e.g., shhs, phys
            logs[f"{n}val_dice"] = dices.mean().round(self.log_round)
            logs[f"{n}val_precision"] = precisions.mean().round(self.log_round)
            logs[f"{n}val_recall"] = recalls.mean().round(self.log_round)
            for m_name, value in metrics[id_].items():
                logs[f"{n}val_{m_name}"] = value.round(self.log_round)

            if self.verbose:
                self._print_val_results(precisions=precisions,
                                        recalls=recalls,
                                        dices=dices,
                                        metrics=metrics[id_],
                                        epoch=epoch,
                                        name=id_,
                                        classes=classes)

        if len(self.IDs) > 1:
            # Print cross-dataset mean values
            if self.verbose:
                self.logger(highlighted(f"[ALL DATASETS] Means Across Classes"
                                        f" for Epoch {epoch}"))
            fetch = ("val_dice", "val_precision", "val_recall")
            m_fetch = tuple(["val_" + s for s in self.model.metrics_names])
            to_print = {}
            for f in fetch + m_fetch:
                scores = [logs["%s_%s" % (name, f)] for name in self.IDs]
                res = np.mean(scores)
                logs[f] = res.round(self.log_round)  # Add to log file
                to_print[f.split("_")[-1]] = list(scores) + [res]
            if self.verbose:
                df = pd.DataFrame(to_print)
                df.index = self.IDs + ["mean"]
                self.logger(df.round(self.print_round))
            self.logger("")


class MemoryConsumption(Callback):
    def __init__(self, max_gib=None, round_=2, logger=None, set_limit=False):
        self.max_gib = max_gib
        self.logger = logger
        self.round_ = round_
        if set_limit:
            import resource
            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS,
                               (self._gib_to_bytes(max_gib), hard))
            self.logger("Setting memory limit to {} GiB".format(max_gib))

    @staticmethod
    def _gib_to_bytes(gib):
        return gib * (1024 ** 3)

    @staticmethod
    def _bytes_to_gib(bytes):
        return bytes / (1024 ** 3)

    def on_epoch_end(self, epoch, logs={}):
        mem_bytes = get_memory_usage()
        mem_gib = round(self._bytes_to_gib(mem_bytes), self.round_)
        logs['memory_usage_gib'] = mem_gib
        if self.max_gib and mem_gib >= self.max_gib:
            self.logger.warn("Stopping training from callback 'MemoryConsumption'! "
                             "Total memory consumption of {} GiB exceeds limitation"
                             " (self.max_gib = {}) ".format(mem_gib, self.max_gib))
            self.model.stop_training = True


class MaxTrainingTime(Callback):
    def __init__(self, max_minutes, log_name='train_time_total', logger=None):
        """
        TODO
        Args:
        """
        super().__init__()
        self.max_minutes = int(max_minutes)
        self.log_name = log_name
        self.logger = logger or ScreenLogger()

    def on_epoch_end(self, epochs, logs={}):
        """
        TODO
        Args:
            epochs:
            logs:
        Returns:
        """
        train_time_str = logs.get(self.log_name, None)
        if not train_time_str:
            self.logger.warn("Did not find log entry '{}' (needed in callback "
                             "'MaxTrainingTime')".format(self.log_name))
            return
        train_time_m = timedelta(
            days=int(train_time_str[:2]),
            hours=int(train_time_str[4:6]),
            minutes=int(train_time_str[8:10]),
            seconds=int(train_time_str[12:14])
        ).total_seconds() / 60
        if train_time_m >= self.max_minutes:
            # Stop training
            self.warn("Stopping training from callback 'MaxTrainingTime'! "
                      "Total training length of {} minutes exceeded (now {}) "
                      "".format(self.max_minutes, train_time_m))
            self.model.stop_training = True


class CarbonUsageTracking(Callback):
    """
    tf.keras Callback for the Carbontracker package.
    See https://github.com/lfwa/carbontracker.
    """
    def __init__(self, epochs, add_to_logs=True, monitor_epochs=-1,
                 epochs_before_pred=-1, devices_by_pid=True, **additional_tracker_kwargs):
        """
        Accepts parameters as per CarbonTracker.__init__
        Sets other default values for key parameters.
        Args:
            add_to_logs: bool, Add total_energy_kwh and total_co2_g to the keras logs after each epoch
            For other arguments, please refer to CarbonTracker.__init__
        """
        super().__init__()
        self.tracker = None
        self.add_to_logs = bool(add_to_logs)
        self.parameters = {"epochs": epochs,
                           "monitor_epochs": monitor_epochs,
                           "epochs_before_pred": epochs_before_pred,
                           "devices_by_pid": devices_by_pid}
        self.parameters.update(additional_tracker_kwargs)

    def on_train_end(self, logs={}):
        """ Ensure actual consumption is reported """
        self.tracker.stop()

    def on_epoch_begin(self, epoch, logs={}):
        """ Start tracking this epoch """
        if self.tracker is None:
            # At this point all CPUs should be discoverable
            self.tracker = CarbonTracker(**self.parameters)
        self.tracker.epoch_start()

    def on_epoch_end(self, epoch, logs={}):
        """ End tracking this epoch """
        self.tracker.epoch_end()
        if self.add_to_logs:
            energy_kwh = self.tracker.tracker.total_energy_per_epoch().sum()
            co2eq_g = self.tracker._co2eq(energy_kwh)
            logs["total_energy_kwh"] = round(energy_kwh, 6)
            logs["total_co2_g"] = round(co2eq_g, 6)

#%% ljy加 - 20211017
class SavePredictionImages(Callback):
    pass
    # """
    # Save images after each epoch of training of the model on a batch of
    # training and a batch of validation data sampled from sequence objects.
    #
    # Saves the input image with ground truth overlay as well as the predicted
    # label masks.
    # """
    # def __init__(self, train_data, val_data, outdir='images'):
    #     """
    #     Args:
    #         train_data: A mpunet.sequence object from which training
    #                     data can be sampled via the __getitem__ method.
    #         val_data:   A mpunet.sequence object from which validation
    #                     data can be sampled via the __getitem__ method.
    #         outdir:     Path to directory (existing or non-existing) in which
    #                     images will be stored.
    #     """
    #     super().__init__()
    #
    #     self.train_data = train_data
    #     self.val_data = val_data
    #     self.save_path = os.path.abspath(os.path.join(outdir, "pred_images_at_epoch"))
    #
    #     if not os.path.exists(self.save_path):
    #         os.makedirs(self.save_path)
    #
    # def pred_and_save(self, data, subdir):
    #     # Get a random batch
    #     X, y = data[np.random.randint(len(data))]
    #
    #     # Predict on the batch
    #     pred = self.model.predict(X)
    #
    #     # print(f'---      ljy: data: {data}' )
    #     # print(f'---             X : {X}' )
    #     # print(f'---             y : {y}' )
    #     # print(f'---           pred: {pred}' )
    #
    #
    #     subdir = os.path.join(self.save_path, subdir)
    #     if not os.path.exists(subdir):
    #         os.mkdir(subdir)
    #
    #     # Plot each sample in the batch
    #     for i, (im, lab, p) in enumerate(zip(X, y, pred)):
    #         fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 6))
    #         lab = lab.reshape(im.shape[:-1] + (lab.shape[-1],))
    #         p = p.reshape(im.shape[:-1] + (p.shape[-1],))
    #         # Imshow ground truth on ax2
    #         # This function will determine which channel, axis and slice to
    #         # show and return so that we can use them for the other 2 axes
    #         chnl, axis, slice = imshow_with_label_overlay(ax2, im, lab, lab_alpha=1.0)
    #
    #         # Imshow pred on ax3
    #         imshow_with_label_overlay(ax3, im, p, lab_alpha=1.0,
    #                                   channel=chnl, axis=axis, slice=slice)
    #
    #         # Imshow raw image on ax1
    #         # Chose the same slice, channel and axis as above
    #         im = im[..., chnl]
    #         im = np.moveaxis(im, axis, 0)
    #         if slice is not None:
    #             # Only for 3D imges
    #             im = im[slice]
    #         ax1.imshow(im, cmap="gray")
    #
    #         # Set labels
    #         ax1.set_title("Image", size=18)
    #         ax2.set_title("True labels", size=18)
    #         ax3.set_title("Prediction", size=18)
    #
    #         fig.tight_layout()
    #         with np.testing.suppress_warnings() as sup:
    #             sup.filter(UserWarning)
    #             fig.savefig(os.path.join(subdir, str(i) + ".png"))
    #         plt.close(fig.number)
    #
    # def on_epoch_end(self, epoch, logs={}):
    #     self.pred_and_save(self.train_data, "train_%s" % epoch)
    #     if self.val_data is not None:
    #         self.pred_and_save(self.val_data, "val_%s" % epoch)
    #


class SaveOutputAs2DImage(Callback):
    pass
#     """
#     Save random 2D slices from the output of a given layer during training.
#     """
#     def __init__(self, layer, sequence, model, out_dir, every=10, logger=None):
#         """
#         Args:
#             layer:    A tf.keras layer
#             sequence: A MultiPlanar.sequence object from which batches are
#                       sampled and pushed through the graph to output of layer
#             model:    A tf.keras model object
#             out_dir:  Path to directory (existing or non-existing) in which
#                       images will be stored
#             every:    Perform this operation every 'every' batches
#         """
#         super().__init__()
#         self.every = every
#         self.seq = sequence
#         self.layer = layer
#         self.epoch = None
#         self.model = model
#         self.logger = logger or ScreenLogger()
#
#         self.out_dir = out_dir
#         if not os.path.exists(out_dir):
#             os.makedirs(self.out_dir)
#         self.log()
#
#     def log(self):
#         self.logger("Save Output as 2D Image Callback")
#         self.logger("Layer:      ", self.layer)
#         self.logger("Every:      ", self.every)
#
#     def on_epoch_begin(self, epoch, logs=None):
#         self.epoch = epoch
#
#     def on_batch_end(self, batch, logs=None):
#         if batch % self.every:
#             return
#
#         # Get output of layer
#         self.model.predict_on_batch()
#         sess = tf.keras.backend.get_session()
#         X, _, _ = self.seq[0]
#         outs = sess.run([self.layer.output], feed_dict={self.model.input: X})[0]
#         if isinstance(outs, list):
#             outs = outs[0]
#
#         for i, (model_in, layer_out) in enumerate(zip(X, outs)):
#             fig = plt.figure(figsize=(12, 6))
#             ax1 = fig.add_subplot(121)
#             ax2 = fig.add_subplot(122)
#
#             # Plot model input and layer outputs on each ax
#             chl1, axis, slice = imshow(ax1, model_in)
#             chl2, _, _ = imshow(ax2, layer_out, axis=axis, slice=slice)
#
#             # Set labels and save figure
#             ax1.set_title("Model input - Channel %i - Axis %i - Slice %i"
#                           % (chl1, axis,slice), size=22)
#             ax2.set_title("Layer output - Channel %i - Axis %i - Slice %i"
#                           % (chl2, axis, slice), size=22)
#
#             fig.tight_layout()
#             fig.savefig(os.path.join(self.out_dir, "epoch_%i_batch_%i_im_%i" %
#                                      (self.epoch, batch, i)))
#             plt.close(fig)