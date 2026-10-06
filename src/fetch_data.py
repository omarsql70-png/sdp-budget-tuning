"""Download the eight PROMISE releases used in the paper and verify their SHA-256 checksums.

Sources (pinned commits, public mirrors of the PROMISE repository):
  * feiwww/PROMISE-backup   (ant-1.7, camel-1.6, xalan-2.4, xerces-1.2, xerces-1.3)
  * klainfo/DefectData (MIT) (ivy-2.0, prop-6, redaktor)
"""
import hashlib, os, sys, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('SDP_DATA', os.path.join(HERE, '..', 'data'))
PB = 'https://raw.githubusercontent.com/feiwww/PROMISE-backup/1e8fd48cbd5275dd57409f3917de4c8ae9e15add/bug-data'
DD = 'https://raw.githubusercontent.com/klainfo/DefectData/e65993d4bf3de34960a80cf1cc98c07224608789/inst/extdata/terapromise/ck'
FILES = {
    'ant-1.7':    (f'{PB}/ant/ant-1.7.csv',       '073f0aaace855d0beb739f6897313d15d68ac11b3057d226b2b4209566ee0a2c'),
    'camel-1.6':  (f'{PB}/camel/camel-1.6.csv',   'e7e0f2fd9f8d6a45dd2cc55ca554cb69151024d4662160d4e7e84a9290b55bdc'),
    'xalan-2.4':  (f'{PB}/xalan/xalan-2.4.csv',   '4ed83c21d32673595dc2b298f137d01081c7b40e66a85246ad5daced208719fb'),
    'xerces-1.2': (f'{PB}/xerces/xerces-1.2.csv', 'b88c9bf9ffd9396644d1057d03456c26e2c50e9accac8ba01ef683155776b40d'),
    'xerces-1.3': (f'{PB}/xerces/xerces-1.3.csv', '9fe65a9e2ae2af31339b9173a2482690e86014c3e6cff34cc50e475f33a1cd3a'),
    'ivy-2.0':    (f'{DD}/ivy-2.0.csv',           'd0c1b20820468a0f3da647ef1d098fcd1d0080595d8e3173151e8504b44ce4cc'),
    'prop-6':     (f'{DD}/prop-6.csv',            '002264853d6b59e6d5d27b1f694fdf115a3b8d815d4073f08f37a598a36bbacf'),
    'redaktor':   (f'{DD}/redaktor.csv',          '64ada67c4ddf5c14b5904af2e84531944ab1b1f08c0c9693e305bf95e2c7d419'),
}
os.makedirs(OUT, exist_ok=True); bad = 0
for name, (url, sha) in FILES.items():
    path = os.path.join(OUT, f'{name}.csv')
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    h = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    ok = h == sha; bad += not ok
    print(f'{name:11s} {"OK" if ok else "CHECKSUM MISMATCH"}')
sys.exit(1 if bad else 0)
