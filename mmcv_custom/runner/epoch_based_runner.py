# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import time
import warnings

import torch

import mmcv
from mmcv.runner.checkpoint import save_checkpoint


class EpochBasedRunner(mmcv.runner.EpochBasedRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        try:
            if create_symlink:
                mmcv.symlink(filename, osp.join(out_dir, 'latest.pth'))
        except:
            from shutil import copyfile
            copyfile(filepath, osp.join(out_dir, 'latest.pth'))



    def auto_resume(self, resume_epoch=None):
        linkname = osp.join(self.work_dir, 'latest.pth')
        assert osp.exists(linkname), "{} not find".format(linkname)
        self.logger.info('latest checkpoint found')
        self.resume(linkname)
