package PerlCuda::Native;
use strict;
use warnings;
use Exporter 'import';
use File::Spec;
our @EXPORT_OK = qw(load_function);

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
1;
