#!/usr/bin/env perl
use strict;
use warnings;
use FindBin;
use lib "$FindBin::Bin/lib";
use Test::More;
use PerlCuda::Common qw(make_inputs reference_batch perl_scalar_batch perl_tight_batch);

my @backends = (
    ['perl_scalar', \&perl_scalar_batch],
    ['perl_batch', \&perl_tight_batch],
);
my $ffi_available = eval { require FFI::Platypus; 1 };
if ($ffi_available) {
    require PerlCuda::Native;
    for my $entry (
        ['cpp_ffi',         'blockdist_cpu',  'blockdist_cpu'],
        ['cuda_ffi_fresh',  'blockdist_cuda', 'blockdist_cuda'],
        ['cuda_ffi_cached', 'blockdist_cuda', 'blockdist_cuda_cached'],
    ) {
        my ($name, $library, $symbol) = @$entry;
        my $ext = $^O eq 'MSWin32' ? '.dll' : '.so';
        my $prefix = $^O eq 'MSWin32' ? '' : 'lib';
        if (-f "$FindBin::Bin/build/$prefix$library$ext") {
            push @backends, [$name, PerlCuda::Native::load_function($FindBin::Bin, $library, $symbol)];
        } else {
            diag "Skipping $name: library not built";
        }
    }
} else {
    diag 'FFI::Platypus not installed; testing pure Perl only';
}

for my $size (1, 7, 1000, 1025) {
    my ($x1, $y1, $x2, $y2) = make_inputs($size);
    my @expected = (0) x $size;
    reference_batch($x1, $y1, $x2, $y2, \@expected, $size);
    for my $backend (@backends) {
        my ($name, $fn) = @$backend;
        my @out = (-1) x $size;
        my $status = $fn->($x1, $y1, $x2, $y2, \@out, $size);
        is($status, 0, "$name status for size=$size");
        is_deeply(\@out, \@expected, "$name exact results for size=$size");
    }
}

for my $case (
    [0, 0, 3, 4, 4],
    [-5, 7, 4, -1, 9],
    [10, 10, 10, 10, 0],
    [-100, -100, 100, 80, 200],
) {
    my ($ax, $ay, $bx, $by, $distance) = @$case;
    for my $backend (@backends) {
        my ($name, $fn) = @$backend;
        my @out = (-1);
        my $status = $fn->([$ax], [$ay], [$bx], [$by], \@out, 1);
        is($status, 0, "$name handcrafted status");
        is($out[0], $distance, "$name handcrafted distance ($ax,$ay)->($bx,$by)");
    }
}

my ($x1, $y1, $x2, $y2) = make_inputs(1000);
$x1->[0] = 991;
$y1->[0] = 0;
$x2->[0] = 0;
$y2->[0] = 990;
my @expected = (0) x 1000;
reference_batch($x1, $y1, $x2, $y2, \@expected, 1000);
for my $backend (@backends) {
    my ($name, $fn) = @$backend;
    my @out = (-1) x 1000;
    is($fn->($x1, $y1, $x2, $y2, \@out, 1000), 0, "$name changed-data status");
    is_deeply(\@out, \@expected, "$name changed-data output");
}

my $cuda_lib_path = "$FindBin::Bin/build/" . ($^O eq 'MSWin32' ? 'blockdist_cuda.dll' : 'libblockdist_cuda.so');
if ($ffi_available && -f $cuda_lib_path) {
    PerlCuda::Native::load_function($FindBin::Bin, 'blockdist_cuda', 'blockdist_cuda_release', 'void')->();
}
done_testing();
