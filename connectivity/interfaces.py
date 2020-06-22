#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Additional interfaces required for workflow. To be upstreamed into Nipype.
Shamelessly taken from https://github.com/poldracklab/ds003-post-fMRIPrep-analysis/blob/master/interfaces.py
"""
from nipype.interfaces.fsl.base import FSLCommand, FSLCommandInputSpec
from nipype.interfaces.base import traits, TraitedSpec


class PtoZInputSpec(FSLCommandInputSpec):
    pvalue = traits.Float(0.05, argstr='%f', usedefault=True, position=0,
                          desc='p-value for which the corresponding '
                               'z-statistic should be computed (default 0.05)')
    two_tailed = traits.Bool(argstr='-2', position=1,
                             desc='use 2-tailed conversion (default is '
                                  '1-tailed)')
    resels = traits.Float(argstr='-g %f', position=2,
                          desc='use GRF maximum-height theory instead of '
                               'Gaussian PDF. To enable this option, specify '
                               'the number of resels as the argument. This can '
                               'be estimated using fsl.SmoothEstimate.')


class PtoZOutputSpec(TraitedSpec):
    zstat = traits.Float(
        desc='z-statistic corresponding to specified p-value')


class PtoZ(FSLCommand):
    """Determine the z-value corresponding to an observed p-value."""
    input_spec = PtoZInputSpec
    output_spec = PtoZOutputSpec
    _cmd = 'ptoz'

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
        outputs = self._outputs()
        outputs.zstat = float(runtime.stdout.strip())
        return outputs


class ClusterInputSpec(FSLCommandInputSpec):
    in_file = File(argstr="--in=%s", mandatory=True, exists=True, desc="input volume")
    threshold = traits.Float(
        argstr="--thresh=%.10f", mandatory=True, desc="threshold for input volume"
    )
    out_index_file = traits.Either(
        traits.Bool,
        File,
        argstr="--oindex=%s",
        desc="output of cluster index (in size order)",
        hash_files=False,
    )
    out_threshold_file = traits.Either(
        traits.Bool,
        File,
        argstr="--othresh=%s",
        desc="thresholded image",
        hash_files=False,
    )
    out_localmax_txt_file = traits.Either(
        traits.Bool,
        File,
        argstr="--olmax=%s",
        desc="local maxima text file",
        hash_files=False,
    )
    out_localmax_vol_file = traits.Either(
        traits.Bool,
        File,
        argstr="--olmaxim=%s",
        desc="output of local maxima volume",
        hash_files=False,
    )
    out_size_file = traits.Either(
        traits.Bool,
        File,
        argstr="--osize=%s",
        desc="filename for output of size image",
        hash_files=False,
    )
    out_max_file = traits.Either(
        traits.Bool,
        File,
        argstr="--omax=%s",
        desc="filename for output of max image",
        hash_files=False,
    )
    out_mean_file = traits.Either(
        traits.Bool,
        File,
        argstr="--omean=%s",
        desc="filename for output of mean image",
        hash_files=False,
    )
    out_pval_file = traits.Either(
        traits.Bool,
        File,
        argstr="--opvals=%s",
        desc="filename for image output of log pvals",
        hash_files=False,
    )
    pthreshold = traits.Float(
        argstr="--pthresh=%.10f",
        requires=["dlh", "volume"],
        desc="p-threshold for clusters",
    )
    peak_distance = traits.Float(
        argstr="--peakdist=%.10f",
        desc="minimum distance between local maxima/minima, in mm (default 0)",
    )
    cope_file = File(argstr="--cope=%s", desc="cope volume")
    volume = traits.Int(argstr="--volume=%d", desc="number of voxels in the mask")
    dlh = traits.Float(
        argstr="--dlh=%.10f", desc="smoothness estimate = sqrt(det(Lambda))"
    )
    fractional = traits.Bool(
        False,
        usedefault=True,
        argstr="--fractional",
        desc="interprets the threshold as a fraction of the robust range",
    )
    connectivity = traits.Int(
        argstr="--connectivity=%d", desc="the connectivity of voxels (default 26)"
    )
    use_mm = traits.Bool(
        False, usedefault=True, argstr="--mm", desc="use mm, not voxel, coordinates"
    )
    find_min = traits.Bool(
        False, usedefault=True, argstr="--min", desc="find minima instead of maxima"
    )
    no_table = traits.Bool(
        False,
        usedefault=True,
        argstr="--no_table",
        desc="suppresses printing of the table info",
    )
    minclustersize = traits.Bool(
        False,
        usedefault=True,
        argstr="--minclustersize",
        desc="prints out minimum significant cluster size",
    )
    xfm_file = File(
        argstr="--xfm=%s",
        desc=(
            "filename for Linear: input->standard-space "
            "transform. Non-linear: input->highres transform"
        ),
    )
    std_space_file = File(
        argstr="--stdvol=%s", desc="filename for standard-space volume"
    )
    num_maxima = traits.Int(argstr="--num=%d", desc="no of local maxima to report")
    warpfield_file = File(argstr="--warpvol=%s", desc="file contining warpfield")
    voxthresh= traits.Bool(
        False,
        usedefault=True,
        argstr="--voxthresh",
        desc="voxel-wise thresholding (corrected)",
    )
    resels = traits.Float(
        argstr="--resels=%.10f",
        desc="Size of one resel in voxel units",
    )


class ClusterOutputSpec(TraitedSpec):
    index_file = File(desc="output of cluster index (in size order)")
    threshold_file = File(desc="thresholded image")
    localmax_txt_file = File(desc="local maxima text file")
    localmax_vol_file = File(desc="output of local maxima volume")
    size_file = File(desc="filename for output of size image")
    max_file = File(desc="filename for output of max image")
    mean_file = File(desc="filename for output of mean image")
    pval_file = File(desc="filename for image output of log pvals")


class Cluster(FSLCommand):
    """ Uses FSL cluster to perform clustering on statistical output
    Examples
    --------
    >>> cl = Cluster()
    >>> cl.inputs.threshold = 2.3
    >>> cl.inputs.in_file = 'zstat1.nii.gz'
    >>> cl.inputs.out_localmax_txt_file = 'stats.txt'
    >>> cl.inputs.use_mm = True
    >>> cl.cmdline
    'cluster --in=zstat1.nii.gz --olmax=stats.txt --thresh=2.3000000000 --mm'
    """

    input_spec = ClusterInputSpec
    output_spec = ClusterOutputSpec
    _cmd = "cluster"

    filemap = {
        "out_index_file": "index",
        "out_threshold_file": "threshold",
        "out_localmax_txt_file": "localmax.txt",
        "out_localmax_vol_file": "localmax",
        "out_size_file": "size",
        "out_max_file": "max",
        "out_mean_file": "mean",
        "out_pval_file": "pval",
    }

    def _list_outputs(self):
        outputs = self.output_spec().get()
        for key, suffix in list(self.filemap.items()):
            outkey = key[4:]
            inval = getattr(self.inputs, key)
            if isdefined(inval):
                if isinstance(inval, bool):
                    if inval:
                        change_ext = True
                        if suffix.endswith(".txt"):
                            change_ext = False
                        outputs[outkey] = self._gen_fname(
                            self.inputs.in_file,
                            suffix="_" + suffix,
                            change_ext=change_ext,
                        )
                else:
                    outputs[outkey] = os.path.abspath(inval)
        return outputs

    def _format_arg(self, name, spec, value):
        if name in list(self.filemap.keys()):
            if isinstance(value, bool):
                fname = self._list_outputs()[name[4:]]
            else:
                fname = value
            return spec.argstr % fname
        return super(Cluster, self)._format_arg(name, spec, value)
