package PerlCuda::Common;
use strict;
use warnings;
use Exporter 'import';
use Getopt::Long qw(GetOptionsFromArray);
use Time::HiRes qw(time);

our @EXPORT_OK = qw(make_inputs reference_batch perl_scalar_batch perl_tight_batch run_benchmark);

# Deterministic coordinates; differences fit signed 32-bit arithmetic.
sub make_inputs {
    my ($n) = @_;
    my (@x1, @y1, @x2, @y2);
    for my $i (0 .. $n - 1) {
        push @x1, ($i * 37 + 11) % 997;
        push @y1, ($i * 79 + 19) % 991;
        push @x2, ($i * 131 + 23) % 983;
        push @y2, ($i * 53 + 29) % 977;
    }
    return (\@x1, \@y1, \@x2, \@y2);
}

sub block_distance {
    my ($x1, $y1, $x2, $y2) = @_;
    my $dx = abs($x2 - $x1);
    my $dy = abs($y2 - $y1);
    return $dx > $dy ? $dx : $dy;
}

sub reference_batch {
    my ($x1, $y1, $x2, $y2, $out, $n) = @_;
    for my $i (0 .. $n - 1) {
        $out->[$i] = block_distance($x1->[$i], $y1->[$i], $x2->[$i], $y2->[$i]);
    }
    return 0;
}

sub perl_scalar_batch { return reference_batch(@_); }

sub perl_tight_batch {
    my ($x1, $y1, $x2, $y2, $out, $n) = @_;
    for (my $i = 0; $i < $n; ++$i) {
        my $dx = abs($x2->[$i] - $x1->[$i]);
        my $dy = abs($y2->[$i] - $y1->[$i]);
        $out->[$i] = $dx > $dy ? $dx : $dy;
    }
    return 0;
}

sub run_benchmark {
    my ($name, $backend, $cleanup) = @_;
    my @argv = @ARGV;
    my ($n, $repeats, $csv) = (1000, 1000, 0);
    GetOptionsFromArray(\@argv,
        'size=i'    => \$n,
        'repeats=i' => \$repeats,
        'csv'       => \$csv,
    ) or die "Usage: perl $0 [--size 1000] [--repeats 1000] [--csv]\n";
    die "Unexpected arguments: @argv\n" if @argv;
    die "--size must be 1..1000000\n" unless $n >= 1 && $n <= 1_000_000;
    die "--repeats must be 1..10000000\n" unless $repeats >= 1 && $repeats <= 10_000_000;

    my ($x1, $y1, $x2, $y2) = make_inputs($n);
    my @expected = (0) x $n;
    my @out = (0) x $n;
    reference_batch($x1, $y1, $x2, $y2, \@expected, $n);

    my $status = $backend->($x1, $y1, $x2, $y2, \@out, $n);
    die "$name: backend error $status\n" if $status != 0;
    for my $i (0 .. $n - 1) {
        die "$name: incorrect output at index $i: got $out[$i], expected $expected[$i]\n"
            if $out[$i] != $expected[$i];
    }

    # Skip initialization/warm-up costs in the timed region.
    for (1 .. 5) {
        $status = $backend->($x1, $y1, $x2, $y2, \@out, $n);
        die "$name: warm-up error $status\n" if $status != 0;
    }

    my $checksum = 0;
    my $start = time();
    for (my $r = 0; $r < $repeats; $r++) {
        $status = $backend->($x1, $y1, $x2, $y2, \@out, $n);
        die "$name: backend error $status at repetition $r\n" if $status != 0;
        $checksum += $out[$r % $n];
    }
    my $seconds = time() - $start;
    $cleanup->() if $cleanup;
    my $total = $n * $repeats;
    my $us_batch = $seconds * 1_000_000 / $repeats;
    my $ns_distance = $seconds * 1_000_000_000 / $total;

    if ($csv) {
        printf "%s,%d,%d,%.9f,%.3f,%.3f,%d\n",
            $name, $n, $repeats, $seconds, $us_batch, $ns_distance, $checksum;
    } else {
        printf "%-19s %7d per batch x %-7d batches = %-10d distances\n",
            $name, $n, $repeats, $total;
        printf "elapsed %.6f s | %.3f us/batch | %.3f ns/distance | checksum %d\n",
            $seconds, $us_batch, $ns_distance, $checksum;
    }
    return 1;
}
1;
