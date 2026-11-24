#!/usr/bin/env perl
use strict;
use warnings;
use FindBin;
use lib "$FindBin::Bin/lib";
use PerlCuda::Packed qw(run_packed_benchmark);
run_packed_benchmark('blockdist_cpu', 'blockdist_cpu', 'cpp', 'prepacked', $FindBin::Bin);
