"""Inspect the native BLAS dependency chain; does not certify a complete wheel."""

from pathlib import Path
import re
import shlex
import subprocess
import tempfile


def run(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def main():
    print(Path('/etc/os-release').read_text(), flush=True)
    print(run('getconf', 'GNU_LIBC_VERSION'))
    print(run('cc', '--version').splitlines()[0])
    print('OpenBLAS version:', run('pkg-config', '--modversion', 'openblas'))
    flags = run('pkg-config', '--cflags', '--libs', 'openblas')
    print('OpenBLAS flags:', flags, flush=True)
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / 'probe.c'
        binary = Path(temporary) / 'probe'
        source.write_text('''#include <cblas.h>
#include <stdio.h>
int main(void) {
    float a[] = {1, 2, 3, 4}, b[] = {5, 6, 7, 8}, c[4] = {0};
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                2, 2, 2, 1, a, 2, b, 2, 0, c, 2);
    printf("SGEMM: %.0f %.0f %.0f %.0f\\n", c[0], c[1], c[2], c[3]);
    return !(c[0] == 19 && c[1] == 22 && c[2] == 43 && c[3] == 50);
}
''')
        print(run('cc', str(source), '-o', str(binary), *shlex.split(flags)))
        print(run(str(binary)))
        dependencies = run('ldd', str(binary))
        print('\nDependency resolution:\n' + dependencies, flush=True)
        libraries = {Path(p).resolve() for p in re.findall(r'=> (/\S+)', dependencies)}
        assert any('openblas' in p.name for p in libraries), 'OpenBLAS not dynamically linked'
        # glibc is supplied by the destination system, not bundled into a wheel.
        system_glibc = {'libc.so.6', 'libm.so.6', 'libpthread.so.0', 'libdl.so.2',
                        'librt.so.1', 'libresolv.so.2', 'libutil.so.1'}
        checked = {binary} | {p for p in libraries if p.name not in system_glibc}
        all_versions = set()
        print('\nUndefined GLIBC symbol requirements (probe + non-glibc dependencies):')
        for library in sorted(checked):
            symbols = run('objdump', '-T', str(library))
            undefined = '\n'.join(line for line in symbols.splitlines() if '*UND*' in line)
            versions = set(re.findall(r'GLIBC_(\d+\.\d+(?:\.\d+)?)', undefined))
            all_versions.update(versions)
            highest = max(versions, key=version_tuple) if versions else 'none'
            print(f'{library}: highest GLIBC_{highest}')
            dynamic = run('readelf', '-d', str(library))
            needed = re.findall(r'\(NEEDED\).*?\[(.*?)\]', dynamic)
            print('  NEEDED:', ', '.join(needed))
        highest = max(all_versions, key=version_tuple)
        print(f'\nObserved GLIBC floor: {highest}')
        for candidate in ('2.28', '2.34', '2.35'):
            compatible = version_tuple(highest) <= version_tuple(candidate)
            print(f'manylinux_{candidate.replace(".", "_")}_x86_64: '
                  f'{"PASS" if compatible else "FAIL"} (GLIBC symbols only)')
        print('\nThis checks native GLIBC requirements only. Audit the final wheel for '
              'other symbol namespaces, ISA requirements, and bundled dependencies.')


def version_tuple(version):
    return tuple(map(int, version.split('.')))


if __name__ == '__main__':
    main()
