import time
from datetime import datetime

import urllib.request

from tqdm import tqdm
import os
from os.path import exists
import glob
import tarfile
import re
import numpy as np

AA_DICT = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
            'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
            'Y': 19}
AA_DICT_WITH_UNKNOWN = AA_DICT.copy()
AA_DICT_WITH_UNKNOWN['-'] = 20

AA_DICT_LONG = {'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4, 'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
            'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14, 'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18,
            'TYR': 19, 'HIE': 6, 'CYX': 1}

AA_DICT_LONG_WITH_UNKNOWN = AA_DICT_LONG.copy()
AA_DICT_LONG_WITH_UNKNOWN['-'] = 20

ATOMIC_DICT = {'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B':5}
DSSP_DICT = {'L': '0', 'H': '1', 'B': '2', 'E': '3', 'G': '4', 'I': '5', 'T': '6', 'S': '7'}
MASK_DICT = {'-': '0', '+': '1'}
NUM_DIMENSIONS = 3



class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)





def prepare_proteinnet(base_location,data_types,version,pnet_filters, output_vars, output_log_units,overwrite_existing=False):
    """
    Wrapper function that downloads and unpacks casp to the specified location
    """
    filename = getproteinnet(base_location,version)
    p_files = unpack_casp_dataset(filename,base_location, data_types)
    files_out = []
    for p_file in p_files:
        path = os.path.dirname(p_file)
        name = os.path.basename(p_file).split(".")[0]
        out_location = path+'/processed_'+name+'/'
        file_out = "{:}/{:}.npz".format(out_location,name)
        if exists(file_out) and not overwrite_existing:
            pass
        else:
            args = parse_pnet(p_file,pnet_filters,output_vars,output_log_units,out_location,overwrite_existing)
            np.savez(file_out, **args)
        # files_out.append(file_out)
    return path, files_out


def getproteinnet(location,version=11):
    """
    This script downloads proteinnet from https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/casp?.tar.gz
    """
    filename = "{:}casp{:}.tar.gz".format(location, version)
    url = 'https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/casp{:}.tar.gz'.format(version)
    if exists(filename):
        print("{:} already exists, skipping download".format(filename))
        return filename
    try:
        print(f"Downloading CASP{version} to {filename}")
        download_url(url, filename)
        # f = wget.download(url, bar=bar_custom, out=location)
        assert(exists(filename))
    except:
        raise RuntimeError("Download failed. Make sure to clean up any tmp files")
    return filename

def unpack_casp_dataset(file, location, data_types):
    files_existing = glob.glob(location+os.path.basename(file).split('.')[0]+'/*')
    files_existing = [x for x in files_existing if os.path.isfile(x)]
    basefiles_existing = [os.path.basename(x) for x in files_existing]
    if all(x in basefiles_existing for x in data_types):
        print("Desired data_types already exists in {:}.".format(location+os.path.basename(file).split('.')[0]))
        return files_existing

    print("Unpacking {:} to {:}".format(file,location))
    t = tarfile.open(file, 'r')
    for member in t.getmembers():
        tfile = os.path.basename(member.name)
        if tfile in data_types:
            if exists(location+'{:}/{:}'.format(os.path.basename(file).split('.')[0],tfile)):
                print("{:} already exist in {:}, skipping extraction.".format(tfile,location))
            else:
                t.extract(member,location)
    t.close()
    files = glob.glob(location+os.path.basename(file).split('.')[0]+'/*')
    files = [x for x in files if os.path.isfile(x)]
    print('Unpacked: {:}'.format(files))
    return files



# Routines to read the file
def separate_coords(full_coords, pos):  # pos can be either 0(n_term), 1(calpha), 2(cterm)
    res = []
    for i in range(len(full_coords[0])):
        if i % 3 == pos:
            res.append([full_coords[j][i] for j in range(3)])

    return res


class switch(object):
    """Switch statement for Python, based on recipe from Python Cookbook."""

    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args:  # changed for v1.5
            self.fall = True
            return True
        else:
            return False


def letter_to_num(string, dict_):
    """ Convert string of letters to list of ints """
    patt = re.compile('[' + ''.join(dict_.keys()) + ']')
    num_string = patt.sub(lambda m: str(dict_[m.group(0)]) + ' ', string)
    num = [int(i) for i in num_string.split()]
    return num


def letter_to_bool(string, dict_):
    """ Convert string of letters to list of bools """
    patt = re.compile('[' + ''.join(dict_.keys()) + ']')
    num_string = patt.sub(lambda m: dict_[m.group(0)] + ' ', string)
    num = [bool(int(i)) for i in num_string.split()]
    return num

def flip_multidimensional_list(list_in):  # pos can be either 0(n_term), 1(calpha), 2(cterm)
    list_out = []
    ld = len(list_in)
    for i in range(len(list_in[0])):
        list_out.append([list_in[j][i] for j in range(ld)])
    return list_out


class list2np(object):
    def __init__(self):
        pass

    def __call__(self, *args):
        args_array = ()
        for arg in args:
            args_array += (np.asarray(arg),)
        return args_array

    def __repr__(self):
        return self.__class__.__name__ + '()'


def read_record(file_, num_evo_entries, use_entropy, use_pssm, use_dssp, use_mask, use_coord, AA_DICT, report_iter=1000, min_seq_len=-1, max_seq_len=999999,scaling=1):
    """
    Read all protein records from pnet file.
    Note that pnet files have coordinates saved in picometers, which is not the normal standard.
    """

    id = []
    seq = []
    pssm = []
    entropy = []
    dssp = []
    coord = []
    mask = []
    seq_len = []

    t0 = time.time()
    cnt = 0
    while True:
        next_line = file_.readline()
        for case in switch(next_line):
            if case('[ID]' + '\n'):
                cnt += 1
                id_i = file_.readline()[:-1]
            elif case('[PRIMARY]' + '\n'):
                seq_i = letter_to_num(file_.readline()[:-1], AA_DICT)
                seq_len_i = len(seq_i)
                if seq_len_i <= max_seq_len and seq_len_i >= min_seq_len:
                    seq_ok = True
                    id.append(id_i)
                    seq.append(seq_i)
                    seq_len.append(seq_len_i)
                else:
                    seq_ok = False
                if (cnt + 1) % report_iter == 0:
                    print("Reading sample: {:}, accepted samples {:} Time: {:2.2f} s".format(cnt, len(id), time.time() - t0))
            elif case('[EVOLUTIONARY]' + '\n'):
                evolutionary = []
                for residue in range(num_evo_entries):
                    evolutionary.append([float(step) for step in file_.readline().split()])
                if use_pssm and seq_ok:
                    pssm.append(evolutionary)
                entropy_i = [float(step) for step in file_.readline().split()]
                if use_entropy and seq_ok:
                    entropy.append(entropy_i)
            elif case('[SECONDARY]' + '\n'):
                dssp_i = letter_to_num(file_.readline()[:-1], DSSP_DICT)
                if use_dssp and seq_ok:
                    dssp.append(dssp_i)
            elif case('[TERTIARY]' + '\n'):
                tertiary = []
                for axis in range(NUM_DIMENSIONS):
                    tertiary.append([float(coord)*scaling for coord in file_.readline().split()])
                if use_coord and seq_ok:
                    coord.append(tertiary)
            elif case('[MASK]' + '\n'):
                mask_i = letter_to_bool(file_.readline()[:-1], MASK_DICT)
                if use_mask and seq_ok:
                    mask.append(mask_i)
            elif case(''):
                return id,seq,pssm,entropy,dssp,coord,mask,seq_len,cnt



class ListToNumpy(object):
    def __init__(self):
        pass
    def __call__(self, args):
        args_array = ()
        for arg in args:
            args_array += (np.asarray(arg),)
        return args_array

def get_default_pnet_filters():
    pnet_filters = {'max_nn_dist': 0,
                    'min_nn_dist': 0,
                    'min_ratio_known': 0.5,
                    'remove_sub_proteins': True,
                    'min_seq_len': 20,
                    'max_seq_len': 1000,
                    }
    return pnet_filters

def get_default_pnet_out_vars():
    output_vars = {'coord': True,
                   'entropy': True,
                   'seq': True,
                   'pssm': True,
                   'mask': True,
                   'dssp': False,
                   }
    return output_vars


def parse_pnet(p_file,pnet_filters,output_vars,output_log_units,location,overwrite_existing=True):
    """
    This is a wrapper for the read_record routine, which reads a pnet-file into memory.
    This routine will convert the lists to numpy arrays, and flip their dimensions to be consistent with how data is normally used in a deep neural network.
    Furthermore the routine will specify the log_unit you wish the data in default is -9 which is equal to nanometer. (Pnet data is given in picometer = -12 by standard)
    """

    def sort_samples_according_to_idx(sort_idx, use_entropy, use_pssm, use_dssp, use_mask, use_coord, rCa, rCb, rN, seq,
                                      seq_len, id, entropy, dssp, pssm, mask):
        """
        This routines sorts samples according to a given list of indices
        """
        seq_len = seq_len[sort_idx]
        seq = [seq[i] for i in sort_idx]
        id = [id[i] for i in sort_idx]

        if use_coord:
            rCa = [rCa[i] for i in sort_idx]
            rCb = [rCb[i] for i in sort_idx]
            rN = [rN[i] for i in sort_idx]
        if use_entropy:
            entropy = [entropy[i] for i in sort_idx]
        if use_pssm:
            pssm = [pssm[i] for i in sort_idx]
        if use_dssp:
            dssp = [dssp[i] for i in sort_idx]
        if use_mask:
            mask = [mask[i] for i in sort_idx]
        return rCa, rCb, rN, seq, seq_len, id, entropy, dssp, pssm, mask

    def keep_samples_according_to_idx(idx_to_keep, use_entropy, use_pssm, use_dssp, use_mask, use_coord, rCa, rCb, rN,
                                      seq, seq_len, id, entropy, dssp, pssm, mask):
        """
        This routine keeps/removes samples according to a boolean list
        """
        indices = np.where(idx_to_keep == True)[0]

        seq = [seq[index] for index in indices]
        seq_len = seq_len[indices]
        id = [id[index] for index in indices]

        if use_coord:
            rCa = [rCa[index] for index in indices]
            rCb = [rCb[index] for index in indices]
            rN = [rN[index] for index in indices]
        if use_entropy:
            entropy = [entropy[index] for index in indices]
        if use_pssm:
            pssm = [pssm[index] for index in indices]
        if use_dssp:
            dssp = [dssp[index] for index in indices]
        if use_mask:
            mask = [mask[index] for index in indices]
        return rCa, rCb, rN, seq, seq_len, id, entropy, dssp, pssm, mask

    def array_in_batched(arr, sub_arr):
        """
        We wish to do subarray comparison in numpy.
        Let arr be an array of size ns,l, where ns is number of samples, and l is length
        and subarr an array of length k < l
        We compare to see whether subarr exist contigious anywhere in arr
        """
        ns, l = arr.shape
        k = len(sub_arr)
        idx = np.arange(l - k + 1)[:, None] + np.arange(k)

        comparison = (arr[:, idx] == sub_arr).all(axis=2)
        result = comparison.any()
        if result:
            tmp = np.where(comparison == True)
            samp = tmp[0][0]
            idx = tmp[1][0]
        else:
            samp = -1
            idx = -1
        return result, samp, idx

    def set_output_vars(output_vars):
        use_entropy = output_vars['entropy']
        use_pssm = output_vars['pssm']
        use_dssp = output_vars['dssp']
        use_mask = output_vars['mask']
        use_coord = output_vars['coord']
        use_seq = output_vars['seq']
        return use_entropy,use_pssm,use_dssp,use_mask,use_coord,use_seq

    os.makedirs(location,exist_ok=True)
    print("Parsing Pnet file: {:}".format(p_file))
    LOG = open("{:}logfile.txt".format(location), "w+")
    LOG.write("Current time = {date:%Y-%m-%d_%H_%M_%S} \n".format(date=datetime.now()))
    LOG.write("pnetfile = {:} \n".format(p_file))
    LOG.write("output_folder = {:} \n".format(location))
    LOG.write("Listing all Filters: \n".format())
    for key, value in pnet_filters.items():
        LOG.write("{:30s} : {} \n".format(key, value))
    LOG.write("Listing all output variables: \n".format())
    for key, value in output_vars.items():
        LOG.write("{:30s} : {} \n".format(key, value))


    use_entropy,use_pssm,use_dssp,use_mask,use_coord,use_seq = set_output_vars(output_vars)
    min_seq_len = pnet_filters['min_seq_len']
    max_seq_len = pnet_filters['max_seq_len']
    pnet_log_unit = -12
    scaling = 10.0 ** (pnet_log_unit - output_log_units)
    t0 = time.time()
    with open(p_file, 'r') as f:
        id, seq, pssm, entropy, dssp, coords, mask, seq_len,total_number_of_proteins = read_record(f, 20, AA_DICT=AA_DICT,
                                                                          use_entropy=use_entropy, use_pssm=use_pssm,
                                                                          use_dssp=use_dssp, use_mask=use_mask,
                                                                          use_coord=use_coord, min_seq_len=min_seq_len,
                                                                          max_seq_len=max_seq_len, scaling=scaling)
    print("Reading records complete. Took: {:2.2f} s".format(time.time() - t0))
    n_org = len(seq)
    rCa = []
    rCb = []
    rN = []

    for i in range(len(coords)):  # We transform each of these, since they are inconveniently stored
        #     # Note that we are changing the order of the coordinates, as well as which one is first, since we want Carbon alpha to be the first, Carbon beta to be the second and Nitrogen to be the third
        rCa.append((separate_coords(coords[i], 1)))
        rCb.append((separate_coords(coords[i], 2)))
        rN.append((separate_coords(coords[i], 0)))
    convert = ListToNumpy()
    rCa = convert(rCa)
    rCb = convert(rCb)
    rN = convert(rN)
    seq = convert(seq)
    entropy = convert(entropy)
    pssm = convert(pssm)
    dssp = convert(dssp)
    mask = convert(mask)

    pssm = [np.swapaxes(pssmi,0,1) for pssmi in pssm]

    seq_len = np.array(seq_len)
    sort_idx = np.argsort(seq_len)


    rCa, rCb, rN, seq, seq_len, id, entropy, dssp, pssm, mask = sort_samples_according_to_idx(sort_idx, use_entropy,use_pssm, use_dssp,use_mask, use_coord, rCa, rCb, rN, seq, seq_len, id, entropy, dssp, pssm, mask)
    min_ratio = pnet_filters['min_ratio_known']

    LOG_ratio_problems = 0
    if min_ratio > 0:
        print("Minimum ratio filter used. Removing proteins with a known coordinate ratio of less than: {:}".format(min_ratio))
        assert use_coord, "coordinates are required to have min_ratio larger than zero"
        idx_to_keep = np.ones(len(seq), dtype=bool)
        for i in range(len(rCa)):
            n = rCa[i].shape[0]
            m = np.sum(rCa[i][:, 0] != 0)
            ratio = m / n
            if ratio < min_ratio:
                LOG_ratio_problems += 1
            idx_to_keep[i] = ratio >= min_ratio
        print("Min ratio removed {:} proteins".format(LOG_ratio_problems))
        rCa, rCb, rN, seq, seq_len, id, entropy, dssp, pssm, mask = keep_samples_according_to_idx(idx_to_keep,use_entropy, use_pssm, use_dssp, use_mask, use_coord, rCa, rCb, rN, seq, seq_len, id, entropy, dssp, pssm, mask)


    max_nn_dist = pnet_filters['max_nn_dist']
    min_nn_dist = pnet_filters['min_nn_dist']

    LOG_min_nn_dist_problems = 0
    LOG_max_nn_dist_problems = 0

    if max_nn_dist > 0 or min_nn_dist > 0:
        print("Removing proteins with too small or large neighboring distance")
        idx_to_keep = np.ones(len(seq), dtype=bool)
        for i in range(len(rCa)):
            if idx_to_keep[i]:
                rCai = rCa[i]
                m = (rCai[:, 0] != 0).astype(np.float32)
                m2 = np.floor((m[1:] + m[:-1]) / 2.0) < 0.5
                m3 = ~ m2
                drCai = rCai[1:, :] - rCai[:-1, :]
                d = np.sqrt(np.sum(drCai ** 2, axis=1))
                dmin = np.min(d[m3])
                dmax = np.max(d[m3])
                if dmin < min_nn_dist:
                    LOG_min_nn_dist_problems += 1
                    idx_to_keep[i] = False
                elif dmax > max_nn_dist:
                    LOG_max_nn_dist_problems += 1
                    idx_to_keep[i] = False
        print("max_nn_dist filter removed {:} proteins".format(LOG_max_nn_dist_problems))
        print("min_nn_dist filter removed {:} proteins".format(LOG_min_nn_dist_problems))
        rCa, rCb, rN, seq, seq_len, id, entropy, dssp, pssm, mask = keep_samples_according_to_idx(idx_to_keep, use_entropy, use_pssm,use_dssp, use_mask,use_coord, rCa, rCb,rN, seq, seq_len, id,entropy, dssp, pssm,mask)

    remove_sub_proteins = pnet_filters['remove_sub_proteins']
    LOG_sub_proteins = 0
    if remove_sub_proteins:
        print("Removing sub-proteins")
        # We assume that the proteins have been sorted by length
        tt0 = time.time()
        n = len(seq)
        idx_to_keep = np.ones(n, dtype=bool)
        parent_proteins = [[] for _ in range(n)]

        seqMat = - np.ones((n,seq_len[-1]),dtype=np.int32)
        for i in range(n):
            seqMat[i,:seq_len[i]] = seq[i]

        for i in range(n):
            if (i + 1) % 1000 == 0:
                print("{:} examples took {:2.2f}".format(i + 1, time.time() - tt0))
            seqi = seq[i]
            result, j, idx = array_in_batched(seqMat[i+1:,:],seqi)
            if result:
                idx_to_keep[i] = False
                LOG_sub_proteins += 1
                if False: # Can be activated to save subproteins if desired
                    ni = len(seqi)
                    r1 = torch.from_numpy(rCa[i].T)
                    r2 = torch.from_numpy(rCa[j][idx:idx + ni].T)
                    cutfullprotein(rCa[j].T, idx, idx + ni,
                                   filename="./../results/figures/cut_in_protein_{:}_{:}".format(i, j))
                    dist, r1cr, r2c = compare_coords_under_rot_and_trans(r1, r2)
                    idi = id[i]
                    if dist < 0.01:
                        folder = "{:}".format("./../data/temp_ok/")
                    else:
                        folder = "{:}".format("./../data/temp/")

                    np.savez(file="{:}subprotein_{:}.npz".format(folder, idi), seq=seq[i], rCa=rCa[i].T,
                             rCb=rCb[i].T, rN=rN[i].T, id=id,
                             log_units=log_unit, AA_LIST=AA_LIST)
                    plot_coordcomparison(r1cr.numpy(), r2c.numpy(),
                                         save_results="{:}comparison_{:}_{:}".format(i, j), num=2,
                                         title="distance = {:2.2f}".format(dist))
                if False: #can be used to plot subprotein / parent comparison
                    # if True:
                    ni = len(seqi)
                    r1 = torch.from_numpy(rCa[i].T)
                    # r2 = torch.from_numpy(rCa[j][0:0+ni].T)
                    r2 = torch.from_numpy(rCa[j][idx:idx + ni].T)
                    cutfullprotein(rCa[j].T, idx, idx + ni,
                                   filename="./../results/figures/cut_in_protein_{:}_{:}".format(i, j))
                    dist, r1cr, r2c = compare_coords_under_rot_and_trans(r1, r2)
                    plot_coordcomparison(r1cr.numpy(), r2c.numpy(),
                                         save_results="./../results/figures/comparison_{:}_{:}".format(i, j), num=2,
                                         title="distance = {:2.2f}".format(dist))
                    print("Subprotein found! {:} is a subprotein of {:}, distance={:2.2f}".format(i, j, dist))
                parent_proteins[j].append([idx, idx + len(seqi)])
                break
        print("Subprotein filter removed {:} proteins".format(LOG_sub_proteins))
        rCa, rCb, rN, seq, seq_len, id, entropy, dssp, pssm, mask = keep_samples_according_to_idx(idx_to_keep,use_entropy, use_pssm,use_dssp, use_mask,use_coord, rCa, rCb, rN, seq, seq_len, id, entropy, dssp, pssm, mask)

    n_removed = n_org - len(seq)
    args = {'id': id,
            'seq': seq,
            'seq_len': seq_len,
            }
    if use_coord:
        args['rCa'] = rCa
        args['rCb'] = rCb
        args['rN'] = rN
    if use_entropy:
        args['entropy'] = entropy
    if use_pssm:
        args['pssm'] = pssm
    if use_dssp:
        args['dssp'] = dssp
    if use_mask:
        args['mask'] = mask
    args['log_units'] = output_log_units
    args['AA_DICT'] = AA_DICT

    LOG.write("parsing pnet complete! Took: {:2.2f} s \n".format(time.time() - t0))
    LOG.write("{:} contained {:} proteins. \n".format(p_file,total_number_of_proteins))
    LOG.write("{:} were kept based on seq len. \n".format(n_org))
    LOG.write("{:} were saved after applying all other filters \n".format(len(seq)))
    print("Parsing Pnet complete \n")
    return args


if __name__ == "__main__":
    """
    This file will download the selected dataset, and prepare it with whatever filters are selected and generate a dataset intended for ML.
    """

    path_proteinnet = '/home/tue/data/casp/'
    data_types = ['testing', 'validation','training_90']
    pnet_output_vars = get_default_pnet_out_vars()
    pnet_filters = get_default_pnet_filters()
    output_log_units = -10  # Angstrom
    pnet_filters['min_nn_dist'] = 1
    pnet_filters['max_nn_dist'] = 10
    pnet_filters['pssm'] = True
    pnet_filters['entropy'] = True

    base_folder, data_files = prepare_proteinnet(path_proteinnet, data_types, version=7, pnet_filters=pnet_filters, output_vars=pnet_output_vars, output_log_units=output_log_units,
                                                 overwrite_existing=False)
