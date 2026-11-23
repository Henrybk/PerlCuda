#!/usr/bin/env perl
use strict;
use warnings;
use FindBin;
use lib "$FindBin::Bin/lib";
use PerlCuda::Common qw(run_benchmark);
use PerlCuda::Native qw(load_function);
my $worker = load_function($FindBin::Bin, 'blockdist_cpu', 'blockdist_cpu');
run_benchmark('cpp_ffi', $worker);
