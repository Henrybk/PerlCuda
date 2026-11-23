#!/usr/bin/env perl
use strict;
use warnings;
use FindBin;
use lib "$FindBin::Bin/lib";
use PerlCuda::Common qw(run_benchmark);
use PerlCuda::Native qw(load_function);
# Allocate/release device arrays for every batch.
my $worker = load_function($FindBin::Bin, 'blockdist_cuda', 'blockdist_cuda');
run_benchmark('cuda_ffi_fresh', $worker);
