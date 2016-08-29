% ckwg +29
% Copyright 2016 by Kitware, Inc.
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
%  * Redistributions of source code must retain the above copyright notice,
%    this list of conditions and the following disclaimer.
%
%  * Redistributions in binary form must reproduce the above copyright notice,
%    this list of conditions and the following disclaimer in the documentation
%    and/or other materials provided with the distribution.
%
%  * Neither name of Kitware, Inc. nor the names of any contributors may be used
%    to endorse or promote products derived from this software without specific
%    prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
% ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%

% Sample image filter written in matlab

%
% This file is loaded into a matlab instance for execution. Functions are called
% by the algorithm driver to pass images and retrieve the result.
%
% The whole filter is a collection of functions that are called by the main C++
% algorithm implementation. These functions can communicate via GLOBAL variables or
% other means. Global variables are used in this example.
%
% The matlab filter control flow is as follows:
%

% 1) The file specified in the pipeline config is loaded into the
%    Matlab engine. A typical configuration for a matlab filter is
%    as follows.
%
% # ================================================================
% process filter
%  :: image_object_filter
%   :filter:type   matlab
%   :filter:matlab:program_file     example_filter/example_matlab_filter.m
%   # Specify initial config for the filter
%   # The following line is presented to the matlab script as "a=1;"
%   :filter:matlab:config:a_var          1
%   :filter:matlab:config:border         2
%   :filter:matlab:config:saving         false
%

% 2) The config values are set into the Matlab engine.
%

% 3) The "filter_initialize()" function is called to enable the
%    filter to perform ant initialization after the configuration is
%    set.
%

% 4) The "check_configuration()" function is called to verify the
%    configuration. This call is part of the C++ algorithm pattern,
%    but may not have much value in matlab.
%

% 5) The "image = apply_filter(image)" function is called with an image. This
%    function does the actual image processing and object
%    detection. Refer to the sample apply_filter() function for the output
%    format and protocol.
%

clear all; clc; close all;

global out_image;

                                % Set configurable values
