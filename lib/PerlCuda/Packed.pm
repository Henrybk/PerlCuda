package PerlCuda::Packed;
use strict;
use warnings;
use Exporter 'import';
use Getopt::Long qw(GetOptionsFromArray);
use Time::HiRes qw(time);
use FFI::Platypus::Buffer qw(scalar_to_buffer);
use PerlCuda::Common qw(make_inputs reference_batch);
use PerlCuda::Native qw(load_pointer_function);

our @EXPORT_OK = qw(pack_inputs pointer_args decoded_output run_packed_benchmark);

# 'l<' is explicitly little-endian signed 32-bit, independent of Perl IV width.
# The test workload fits int32 and the native routines use int32_t.
sub pack_inputs {
    my ($x1, $y1, $x2, $y2) = @_;
    return map { pack('l<*', @$_) } ($x1, $y1, $x2, $y2);
}

# Return five raw addresses + int32 count, without FFI array conversion.
# Callers must retain the four packed scalars and output scalar, and must not
# modify/resize them until the synchronous native call has returned.
sub pointer_args {
    my ($packed, $output, $n) = @_;
    die "Expected exactly four input buffers\n" unless @$packed == 4;
    die "Invalid batch size\n" unless defined $n && $n >= 1 && $n <= 1_000_000;
    my $bytes = 4 * $n;
    my @addresses;
    for my $value (@$packed) {
        my ($ptr, $len) = scalar_to_buffer($value);
        die "Incorrect input buffer length ($len vs $bytes)\n" unless $len == $bytes;
        push @addresses, $ptr;
    }
    my ($result_ptr, $result_len) = scalar_to_buffer($$output);
    die "Incorrect output buffer length\n" unless $result_len == $bytes;
    return (@addresses, $result_ptr, $n);
}

sub decoded_output {
    my ($buffer_ref) = @_;
    return [unpack('l<*', $$buffer_ref)];
}

sub run_packed_benchmark {
    my ($library, $symbol, $prefix, $mode, $root) = @_;
    my @argv = @ARGV;
    my ($n, $repeats, $csv) = (1000, 1000, 0);
    GetOptionsFromArray(\@argv,
        'size=i' => \$n, 'repeats=i' => \$repeats, 'csv' => \$csv,
        'mode=s' => \$mode,
    ) or die "Usage: perl $0 [--mode prepacked|end-to-end] [--size N] [--repeats R] [--csv]\n";
    die "Unexpected arguments: @argv\n" if @argv;
    die "Unknown mode '$mode'\n" unless $mode eq 'prepacked' || $mode eq 'end-to-end';
    die "Invalid size\n" unless $n >= 1 && $n <= 1_000_000;
    die "Invalid repeats\n" unless $repeats >= 1 && $repeats <= 10_000_000;

    my $worker = load_pointer_function($root, $library, $symbol);
    my ($x1, $y1, $x2, $y2) = make_inputs($n);
    my @expected = (0) x $n;
    reference_batch($x1, $y1, $x2, $y2, \@expected, $n);

    # PREPACKED: keep packed Perl strings and their addresses unchanged.
    my @packed = pack_inputs($x1, $y1, $x2, $y2);
    my $output = "\0" x (4 * $n);
    my @native_args = pointer_args(\@packed, \$output, $n);
    my $last_decoded;
    my $run = $mode eq 'prepacked'
        ? sub { return $worker->(@native_args); }
        : sub {
            # END-TO-END: pack four fresh input arrays, obtain raw pointers,
            # call native backend and unpack ALL results on every iteration.
            my @fresh = pack_inputs($x1, $y1, $x2, $y2);
            my $fresh_output = "\0" x (4 * $n);
            my @args = pointer_args(\@fresh, \$fresh_output, $n);
            my $rc = $worker->(@args);
            $last_decoded = decoded_output(\$fresh_output) if $rc == 0;
            return $rc;
        };

    my $validate = sub {
        my $status = $run->();
        die "Native error $status\n" unless $status == 0;
        my $actual = $mode eq 'prepacked' ? decoded_output(\$output) : $last_decoded;
        for my $i (0 .. $n - 1) {
            die "Mismatch at $i: $actual->[$i] != $expected[$i]\n"
                if $actual->[$i] != $expected[$i];
        }
    };
    $validate->();
    for (1 .. 5) {
        my $status = $run->();
        die "Warmup error $status\n" unless $status == 0;
    }

    my $checksum = 0;
    my $start = time();
    for (my $r = 0; $r < $repeats; ++$r) {
        my $status = $run->();
        die "Native error $status at batch $r\n" unless $status == 0;
        $checksum += $mode eq 'prepacked'
            ? unpack('l<', substr($output, 4 * ($r % $n), 4))
            : $last_decoded->[$r % $n];
    }
    my $seconds = time() - $start;
    $validate->();
    my $name = $prefix . ($mode eq 'prepacked' ? '_prepacked' : '_packed_e2e');
    if ($csv) {
        printf "%s,%d,%d,%.9f,%.3f,%.3f,%d\n", $name, $n, $repeats,
            $seconds, $seconds * 1e6 / $repeats,
            $seconds * 1e9 / ($repeats * $n), $checksum;
    } else {
        printf "%s | size=%d repeats=%d | elapsed=%.6f s | %.3f us/batch | %.3f ns/distance | checksum=%d\n",
            $name, $n, $repeats, $seconds,
            $seconds * 1e6 / $repeats, $seconds * 1e9 / ($repeats * $n), $checksum;
    }
    return 1;
}
1;
