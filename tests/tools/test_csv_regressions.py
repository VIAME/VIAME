from .tool_regression_helpers import tool_env, run_viame


def test_filter_before_renumber(tool_env, tmp_path):
    p = tmp_path / 'tracks.csv'
    p.write_text('42,a.png,0,0,0,10,10,1,-1,fish,1\n42,b.png,1,0,0,10,10,1,-1,fish,1\n99,c.png,2,0,0,10,10,1,-1,fish,1\n')
    r = run_viame(tool_env, 'csv', '-i', str(p), '--assign-uid', '--filter-single')
    assert r.returncode == 0, r.stderr
    assert [line.split(',')[0] for line in p.read_text().splitlines()] == ['1', '1']


def test_malformed_row_does_not_replace_original(tool_env, tmp_path):
    p = tmp_path / 'tracks.csv'
    text = '42,a.png\n'
    p.write_text(text)
    r = run_viame(tool_env, 'csv', '-i', str(p), '--increase-fid')
    assert r.returncode != 0
    assert p.read_text() == text
