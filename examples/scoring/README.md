Scoring Notes
-------------

The kwant package provides a scoring tool that can be used to
calculate the probability of detecting an item. The input to this tool
must be in the KitWare kw18 format. Several scripts are provided to
convert habcam annotations and Scallop-TK output to kw18 format. The
format is very simple so additional converters can be easily created.

The scoring tool takes two files: the actual detections in the truth
file and the computed detections. The computed detections are scored
against the truth file to give a set of statistics as shown below:

```
    HADWAV Scoring Results:
      Detection-Pd: 0.748387
      Detection-FA: 8
      Detection-PFA: 0.0338983
      Frame-NFAR: not computed
      Track-Pd: 0.748387
      Track-FA: 8
      Computed-track-PFA: 0.0338983
      Track-NFAR: not computed
      Avg track (continuity, purity ): 13.693, 1
      Avg target (continuity, purity ): 20.1419, 0.748387
      Track-frame-precision: 0.947826
```

The tool was originally written to analyze object tracks in full
motion video imagery so some of the terminology and calculated metrics
may not apply.

One main metric is the probability of detection Pd. This is calculated
as follows:

    Pd = (num detections match truth) / (num truth)

Detection files can be written in the kw18 format by using the
appropriate writer in the pipeline or by running one of these
converters. One downside to using the kw18 writer in the pipeline is
that the image file name is not captured.  All the converters take the
same set of command line options. For example:

```
    Usage: habcam_to_kw18.pl [opts] file
      Options:
        --help                     print usage
        --write-file file-name     Write image file/index correspondence to file
        --read-file  file-name     Read image file/index correspondence to file
        --cache-only               With --in-file, does not add process images unless they are already in cache
```

In order to get the best statistics the number of images processed
must be the same as the number of images in the truth set. Computed
detections and truth are compared on an image basis so the number of
truth entries must be limited to the same number of images as the
computed detections. The options to these converters aide in this regard.

Calculated detections are converted first and use the --out-file
option to write out the list of files processed. The truth set is
processed next with the --in-file option referring to the file created
in the previous step. The --cache-only flag should be added to this
second conversion to cause images not in the first step to be skipped.

The score_tracks tool is run as follows:

     score_tracks --computed-tracks computed_det.kw18 --truth-tracks ground_truth2.kw18

A full list of the options can be coaxed from the tool by using the
`-?` option.
