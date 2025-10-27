from jinja2 import Template
from scipy.interpolate import make_smoothing_spline

def get_extrema_pos(X, n_extrema=10):
    s = make_smoothing_spline(X.energy, X.mu, lam=0.1)
    sd = s.derivative(nu=1)
    sd_roots = sd.roots()

    extrema_pos = sd_roots[sd_roots>X.e0][:n_extrema]
    return extrema_pos, s(extrema_pos)


def header_template(cfg, code, region='xanes'):
    template_dir = cfg['dir']['project_root'] + '/' + cfg['dir']['template_dir']

    with open(f'{template_dir}/{code}-{region}-header.j2', 'r') as header_file:
        template_string = header_file.read()
        templ = Template(template_string)

    return templ
