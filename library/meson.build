project('mpi-preload', 'c')

cc = meson.get_compiler('c')
mpic = dependency('mpi', language : 'c', method :'config-tool')
libdl = cc.find_library('dl')
librt = cc.find_library('rt', required: false)

lz4_dep = dependency('liblz4')
zstd_dep = dependency('libzstd')
glib_dep = dependency('glib-2.0')
zlib_dep = dependency('zlib')
hdf5_dep = dependency('hdf5')
omp_dep = dependency('openmp', required: false)

m_dep = cc.find_library('m', required : false)

onnxrt = cc.find_library('onnxruntime', required: true)
onnxrt_dep = declare_dependency(dependencies: onnxrt)

deps = [m_dep, omp_dep, lz4_dep, zstd_dep, glib_dep, zlib_dep, hdf5_dep, onnxrt_dep]

preload_incs = include_directories([
	'include',
])

preload_srcs = files([
    'lib/preload.c',
	'lib/tracing.c',
	'lib/meta.c',
	'lib/util.c',
	'lib/filter.c',
	'lib/settings.c',
	'lib/compression.c',
	'lib/compression/zstd.c',
	'lib/compression/lz4.c',
	'lib/compression/lz4-fast.c',
	'lib/compression/zlib.c',
	'lib/intercept/mpi-io.c',
	'lib/analysis/compression.c',
	'lib/inferencing/compression.c'
])

preload_lib = shared_library('mpi-preload', preload_srcs,
	dependencies: [mpic, libdl, librt, deps],
	include_directories: preload_incs,
	#soversion: meson.project_version().split('.')[0],
	#version: meson.project_version(),
	install: true,
)

ioa_dep = declare_dependency(link_with: preload_lib)
png_dep = dependency('libpng', method: 'pkg-config')

inferencing_demo_srcs = files([
	'tools/inferencing/inferencing.c',
])

inferencing_demo = executable('inferencing-demo', inferencing_demo_srcs,
	dependencies: [ioa_dep, deps, png_dep],
	include_directories: [preload_incs] + [include_directories('tools/inferencing')],
)

inferencing_io_srcs = files([
	'tools/inferencing-io/inferencing-io.c',
])

inferencing_io = executable('inferencing-io', inferencing_io_srcs,
	dependencies: [ioa_dep, mpic, deps],
	include_directories: [preload_incs] + [include_directories('tools/inferencing-io')],
)