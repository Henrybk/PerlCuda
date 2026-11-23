#!/usr/bin/env perl
use strict;
use warnings;
use FindBin;
use lib "$FindBin::Bin/lib";
use PerlCuda::Common qw(perl_scalar_batch run_benchmark);
run_benchmark('perl_scalar', \&perl_scalar_batch);
