#!/usr/bin/env perl
use strict;
use warnings;
use FindBin;
use File::Spec;
use Getopt::Long qw(GetOptions);

my ($size, $repeats) = (1000, 1000);
GetOptions('size=i' => \$size, 'repeats=i' => \$repeats)
    or die "Usage: perl bench_all.pl [--size 1000] [--repeats 1000]\n";
my $ext = $^O eq 'MSWin32' ? 'dll' : 'so';
my $prefix = $^O eq 'MSWin32' ? '' : 'lib';
my $ffi_available = eval { require FFI::Platypus; 1 };
my @cases = (
    ['01_perl_scalar.pl'],
    ['02_perl_batch.pl'],
    ['03_cpp_ffi.pl', join('', $prefix, 'blockdist_cpu.', $ext)],
    ['04_cuda_ffi.pl', join('', $prefix, 'blockdist_cuda.', $ext)],
    ['05_cuda_cached_ffi.pl', join('', $prefix, 'blockdist_cuda.', $ext)],
    ['06_cpp_packed.pl', join('', $prefix, 'blockdist_cpu.', $ext), 'prepacked'],
    ['07_cuda_packed.pl', join('', $prefix, 'blockdist_cuda.', $ext), 'prepacked'],
    ['06_cpp_packed.pl', join('', $prefix, 'blockdist_cpu.', $ext), 'end-to-end'],
    ['07_cuda_packed.pl', join('', $prefix, 'blockdist_cuda.', $ext), 'end-to-end'],
);
print "method,size,repeats,elapsed_seconds,microseconds_per_batch,nanoseconds_per_distance,checksum\n";
for my $case (@cases) {
    my ($script, $lib, $mode) = @$case;
    if ($lib && !$ffi_available) {
        warn "SKIP $script: FFI::Platypus not installed\n";
        next;
    }
    if ($lib && !-f File::Spec->catfile($FindBin::Bin, 'build', $lib)) {
        warn "SKIP $script: build/$lib missing\n";
        next;
    }
    my @cmd = ($^X, File::Spec->catfile($FindBin::Bin, $script),
               '--size', $size, '--repeats', $repeats, '--csv');
    push @cmd, ('--mode', $mode) if defined $mode;
    system(@cmd) == 0 or die "$script failed (exit code $?)\n";
}
