#!/usr/bin/env perl
use strict;
use warnings;
use FindBin;
use lib "$FindBin::Bin/lib";
use Getopt::Long qw(GetOptions);
use Time::HiRes qw(time);
use FFI::Platypus::Buffer qw(scalar_to_buffer);
use PerlCuda::Common qw(make_inputs reference_batch);
use PerlCuda::Native qw(load_pointer_function);
use PerlCuda::Packed qw(pack_inputs pointer_args decoded_output);

my ($n, $repeats) = (1000, 200);
GetOptions('size=i' => \$n, 'repeats=i' => \$repeats)
    or die "Usage: perl $0 [--size N] [--repeats R]\n";
die "Invalid size\n" unless $n >= 1 && $n <= 1_000_000;
die "Invalid repeats\n" unless $repeats >= 1 && $repeats <= 10_000_000;
my ($x1, $y1, $x2, $y2) = make_inputs($n);
my @expected = (0) x $n;
reference_batch($x1, $y1, $x2, $y2, \@expected, $n);
my @packed = pack_inputs($x1, $y1, $x2, $y2);
my $output = "\0" x (4 * $n);
my @args = pointer_args(\@packed, \$output, $n);
my $timings = "\0" x 16; # 4 little-endian float32 values
my ($times_ptr, $times_size) = scalar_to_buffer($timings);
die "Invalid timings buffer\n" unless $times_size == 16;
my $run = load_pointer_function($FindBin::Bin, 'blockdist_cuda',
    'blockdist_cuda_profile_cached', 'profile');
# Argument order: four input pointers, output pointer, n, metrics pointer.
my $status = $run->(@args, $times_ptr);
die "CUDA profile error $status\n" unless $status == 0;
for my $i (0 .. $n - 1) {
    my $got = unpack('l<', substr($output, 4 * $i, 4));
    die "Incorrect result at $i\n" if $got != $expected[$i];
}
my @sum = (0, 0, 0, 0);
my $checksum = 0;
my $begin = time();
for (my $i = 0; $i < $repeats; ++$i) {
    $status = $run->(@args, $times_ptr);
    die "CUDA profile error $status\n" unless $status == 0;
    my @ms = unpack('f<4', $timings);
    $sum[$_] += $ms[$_] for 0 .. 3;
    $checksum += unpack('l<', substr($output, 4 * ($i % $n), 4));
}
my $elapsed = time() - $begin;
print "CUDA packed diagnostic | $n distances/batch | $repeats repeats\n";
printf "Perl wall (includes FFI + instrumentation) : %.3f us/batch\n", 1e6 * $elapsed / $repeats;
printf "H2D transfers (host wall, four copies)    : %.3f us/batch\n", 1e3 * $sum[0] / $repeats;
printf "Kernel (CUDA event GPU elapsed)           : %.3f us/batch\n", 1e3 * $sum[1] / $repeats;
printf "D2H transfer (host wall)                  : %.3f us/batch\n", 1e3 * $sum[2] / $repeats;
printf "Native instrumented wall (with event sync): %.3f us/batch\n", 1e3 * $sum[3] / $repeats;
print "Checksum: $checksum\n";
print "Diagnostic timings contain event overhead; do not subtract them from ordinary benchmarks.\n";
