package PerlCuda::Native;
use strict;
use warnings;
use Exporter 'import';
use File::Spec;
our @EXPORT_OK = qw(load_function load_pointer_function);

# Five int32_t* arrays and int32_t count, C ABI.
sub load_function {
    my ($root, $library, $symbol, $return_type) = @_;
    require FFI::Platypus;
    my $filename = $^O eq 'MSWin32' ? "$library.dll" : "lib$library.so";
    my $path = File::Spec->catfile($root, 'build', $filename);
    die "Missing $path. Build native code first (see README.md).\n" unless -f $path;
    my $ffi = FFI::Platypus->new(api => 2, lib => $path);
    my $fn = $return_type && $return_type eq 'void'
        ? $ffi->function($symbol => [] => 'void')
        : $ffi->function($symbol => [ ('sint32[]') x 5, 'sint32' ] => 'sint32');
    # Retain FFI handle for as long as callback is alive.
    return sub { my $keep_alive = $ffi; return $fn->call(@_); };
}

# Packed int32 buffers: pass opaque pointers, not sint32[] conversions.
# The caller owns and must keep every pointed-to Perl scalar alive for the
# entire synchronous call. "profile" adds a final float[4] output pointer.
sub load_pointer_function {
    my ($root, $library, $symbol, $kind) = @_;
    require FFI::Platypus;
    my $filename = $^O eq 'MSWin32' ? "$library.dll" : "lib$library.so";
    my $path = File::Spec->catfile($root, 'build', $filename);
    die "Missing $path. Build native code first.\n" unless -f $path;
    die "Unknown native signature '$kind'\n"
        if defined($kind) && $kind ne 'profile';
    my @args = ((('opaque') x 5), 'sint32');
    push @args, 'opaque' if defined($kind) && $kind eq 'profile';
    my $ffi = FFI::Platypus->new(api => 2, lib => $path);
    my $method = $ffi->function($symbol => \@args => 'sint32');
    return sub { my $keep_alive = $ffi; return $method->call(@_); };
}

1;
