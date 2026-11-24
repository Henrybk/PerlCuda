#!/usr/bin/env perl
use strict;
use warnings;
use FindBin;
use lib "$FindBin::Bin/lib";
use Test::More;
use PerlCuda::Common qw(make_inputs reference_batch);

my $have_ffi = eval {
    require FFI::Platypus;
    require FFI::Platypus::Buffer;
    require PerlCuda::Packed;
    require PerlCuda::Native;
    1;
};
if (!$have_ffi) {
    plan skip_all => "Packed tests need FFI::Platypus and its Buffer module: $@";
}
my @candidates = (
    ['cpp_packed', 'blockdist_cpu', 'blockdist_cpu'],
    ['cuda_packed', 'blockdist_cuda', 'blockdist_cuda_cached'],
    ['cuda_profile', 'blockdist_cuda', 'blockdist_cuda_profile_cached', 'profile'],
);
my $ext = $^O eq 'MSWin32' ? '.dll' : '.so';
my $prefix = $^O eq 'MSWin32' ? '' : 'lib';
my @active;
for my $candidate (@candidates) {
    my ($name, $library, $symbol, $kind) = @$candidate;
    my $filename = "$FindBin::Bin/build/$prefix$library$ext";
    if (-f $filename) {
        push @active, [$name, PerlCuda::Native::load_pointer_function(
            $FindBin::Bin, $library, $symbol, $kind), $kind];
    } else {
        diag "Skipping $name; missing $filename";
    }
}
for my $size (1, 7, 1000, 1025, 10000) {
    my ($x1, $y1, $x2, $y2) = make_inputs($size);
    my @reference = (0) x $size;
    reference_batch($x1, $y1, $x2, $y2, \@reference, $size);
    my @packed = PerlCuda::Packed::pack_inputs($x1, $y1, $x2, $y2);
    for my $i (0 .. 3) {
        is(length($packed[$i]), 4 * $size, "input buffer size $i at n=$size");
    }
    for my $candidate (@active) {
        my ($name, $call, $kind) = @$candidate;
        my $out = "\0" x (4 * $size);
        my @args = PerlCuda::Packed::pointer_args(\@packed, \$out, $size);
        my $metrics = "\0" x 16;
        my ($metrics_ptr, $metrics_bytes) =
            FFI::Platypus::Buffer::scalar_to_buffer($metrics);
        push @args, $metrics_ptr if $kind && $kind eq 'profile';
        is($call->(@args), 0, "$name succeeded at n=$size");
        is_deeply(PerlCuda::Packed::decoded_output(\$out), \@reference,
            "$name returned exact distances at n=$size");
        is(length($metrics), 16, "$name preserved timing buffer length")
            if $kind && $kind eq 'profile';
        is($call->(@args), 0, "$name reused buffers at n=$size");
        is_deeply(PerlCuda::Packed::decoded_output(\$out), \@reference,
            "$name reused buffers correctly at n=$size");
    }
}
for my $candidate (@active) {
    my ($name, $call, $kind) = @$candidate;
    my @x1 = (-100, 20, 5, 0);
    my @y1 = (9, -1, 5, 0);
    my @x2 = (1, -20, 5, -1);
    my @y2 = (-200, 100, 5, 0);
    for my $iteration (1 .. 3) {
        $x1[0] -= $iteration;
        $y2[2] += $iteration;
        my @expected = (0) x 4;
        reference_batch(\@x1, \@y1, \@x2, \@y2, \@expected, 4);
        my @packed = PerlCuda::Packed::pack_inputs(\@x1, \@y1, \@x2, \@y2);
        my $out = "\0" x 16;
        my @args = PerlCuda::Packed::pointer_args(\@packed, \$out, 4);
        my $metrics = "\0" x 16;
        my ($timing_ptr) = FFI::Platypus::Buffer::scalar_to_buffer($metrics);
        push @args, $timing_ptr if $kind && $kind eq 'profile';
        is($call->(@args), 0, "$name modified batch $iteration status");
        is_deeply(PerlCuda::Packed::decoded_output(\$out), \@expected,
            "$name modified batch $iteration exact outputs");
    }
}
done_testing();
