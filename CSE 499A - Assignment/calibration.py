

from typing import List, Tuple, Union
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, QiskitError
from qiskit.result import Result
import copy
import re
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
import scipy.linalg as la
from qiskit.utils import parallel_map
import qiskit

try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False



def complete_meas_cal(qubit_list: List[int] = None,
                      qr: Union[int, List[QuantumRegister]] = None,
                      cr: Union[int, List[ClassicalRegister]] = None,
                      circlabel: str = ''
                      ) -> Tuple[List[QuantumCircuit], List[str]
                                 ]:
    """
    Return a list of measurement calibration circuits for the full
    Hilbert space.

    If the circuit contains :math:`n` qubits, then :math:`2^n` calibration circuits
    are created, each of which creates a basis state.

    Args:
        qubit_list: A list of qubits to perform the measurement correction on.
           If `None`, and qr is given then assumed to be performed over the entire
           qr. The calibration states will be labelled according to this ordering (default `None`).

        qr: Quantum registers (or their size).
        If `None`, one is created (default `None`).

        cr: Classical registers (or their size).
        If `None`, one is created(default `None`).

        circlabel: A string to add to the front of circuit names for
            unique identification(default ' ').

    Returns:
        A list of QuantumCircuit objects containing the calibration circuits.

        A list of calibration state labels.

    Additional Information:
        The returned circuits are named circlabel+cal_XXX
        where XXX is the basis state,
        e.g., cal_1001.

        Pass the results of these circuits to the CompleteMeasurementFitter
        constructor.

    Raises:
        QiskitError: if both `qubit_list` and `qr` are `None`.

    """

    if qubit_list is None and qr is None:
        raise QiskitError("Must give one of a qubit_list or a qr")

    # Create the registers if not already done
    if qr is None:
        qr = QuantumRegister(max(qubit_list)+1)

    if isinstance(qr, int):
        qr = QuantumRegister(qr)

    if qubit_list is None:
        qubit_list = range(len(qr))

    if isinstance(cr, int):
        cr = ClassicalRegister(cr)

    nqubits = len(qubit_list)

    # labels for 2**n qubit states
    state_labels = count_keys(nqubits)

    cal_circuits, _ = tensored_meas_cal([qubit_list],
                                        qr, cr, circlabel)

    return cal_circuits, state_labels




def count_keys(num_qubits: int) -> List[str]:
    """Return ordered count keys.

    Args:
        num_qubits: The number of qubits in the generated list.
    Returns:
        The strings of all 0/1 combinations of the given number of qubits
    Example:
        >>> count_keys(3)
        ['000', '001', '010', '011', '100', '101', '110', '111']
    """
    return [bin(j)[2:].zfill(num_qubits)
            for j in range(2 ** num_qubits)]


def tensored_meas_cal(mit_pattern: List[List[int]] = None,
                      qr: Union[int, List[QuantumRegister]] = None,
                      cr: Union[int, List[ClassicalRegister]] = None,
                      circlabel: str = ''
                      ) -> Tuple[List[QuantumCircuit], List[List[int]]
                                 ]:
    """
    Return a list of calibration circuits

    Args:
        mit_pattern: Qubits on which to perform the
            measurement correction, divided to groups according to tensors.
            If `None` and `qr` is given then assumed to be performed over the entire
            `qr` as one group (default `None`).

        qr: A quantum register (or its size).
        If `None`, one is created (default `None`).

        cr: A classical register (or its size).
        If `None`, one is created (default `None`).

        circlabel: A string to add to the front of circuit names for
            unique identification (default ' ').

    Returns:
        A list of two QuantumCircuit objects containing the calibration circuits
        mit_pattern

    Additional Information:
        The returned circuits are named circlabel+cal_XXX
        where XXX is the basis state,
        e.g., cal_000 and cal_111.

        Pass the results of these circuits to the TensoredMeasurementFitter
        constructor.

    Raises:
        QiskitError: if both `mit_pattern` and `qr` are None.
        QiskitError: if a qubit appears more than once in `mit_pattern`.

    """

    if mit_pattern is None and qr is None:
        raise QiskitError("Must give one of mit_pattern or qr")

    if isinstance(qr, int):
        qr = QuantumRegister(qr)

    qubits_in_pattern = []
    if mit_pattern is not None:
        for qubit_list in mit_pattern:
            for qubit in qubit_list:
                if qubit in qubits_in_pattern:
                    raise QiskitError("mit_pattern cannot contain \
                    multiple instances of the same qubit")
                qubits_in_pattern.append(qubit)

        # Create the registers if not already done
        if qr is None:
            qr = QuantumRegister(max(qubits_in_pattern)+1)
    else:
        qubits_in_pattern = range(len(qr))
        mit_pattern = [qubits_in_pattern]

    nqubits = len(qubits_in_pattern)

    # create classical bit registers
    if cr is None:
        cr = ClassicalRegister(nqubits)

    if isinstance(cr, int):
        cr = ClassicalRegister(cr)

    qubits_list_sizes = [len(qubit_list) for qubit_list in mit_pattern]
    nqubits = sum(qubits_list_sizes)
    size_of_largest_group = max(qubits_list_sizes)
    largest_labels = count_keys(size_of_largest_group)

    state_labels = []
    for largest_state in largest_labels:
        basis_state = ''
        for list_size in qubits_list_sizes:
            basis_state = largest_state[:list_size] + basis_state
        state_labels.append(basis_state)

    cal_circuits = []
    for basis_state in state_labels:
        qc_circuit = QuantumCircuit(qr, cr,
                                    name='%scal_%s' % (circlabel, basis_state))

        end_index = nqubits
        for qubit_list, list_size in zip(mit_pattern, qubits_list_sizes):

            start_index = end_index - list_size
            substate = basis_state[start_index:end_index]

            for qind in range(list_size):
                if substate[list_size-qind-1] == '1':
                    qc_circuit.x(qr[qubit_list[qind]])

            end_index = start_index

        qc_circuit.barrier(qr)

        # add measurements
        
        qc_circuit.measure_all()

        # end_index = nqubits
        # for qubit_list, list_size in zip(mit_pattern, qubits_list_sizes):

        #     for qind in range(list_size):
        #         qc_circuit.measure(qr[qubit_list[qind]],
        #                            cr[nqubits-(end_index-qind)])

        #     end_index -= list_size

        cal_circuits.append(qc_circuit)

    return cal_circuits, mit_pattern











class CompleteMeasFitter:
    """
    Measurement correction fitter for a full calibration
    """

    def __init__(self,
                 results: Union[Result, List[Result]],
                 state_labels: List[str],
                 qubit_list: List[int] = None,
                 circlabel: str = ''):
        """
        Initialize a measurement calibration matrix from the results of running
        the circuits returned by `measurement_calibration_circuits`

        A wrapper for the tensored fitter

        Args:
            results: the results of running the measurement calibration
                circuits. If this is `None` the user will set a calibration
                matrix later.
            state_labels: list of calibration state labels
                returned from `measurement_calibration_circuits`.
                The output matrix will obey this ordering.
            qubit_list: List of the qubits (for reference and if the
                subset is needed). If `None`, the qubit_list will be
                created according to the length of state_labels[0].
            circlabel: if the qubits were labeled.
        """
        if qubit_list is None:
            qubit_list = range(len(state_labels[0]))
        self._qubit_list = qubit_list

        self._tens_fitt = TensoredMeasFitter(results,
                                             [qubit_list],
                                             [state_labels],
                                             circlabel)

    @property
    def cal_matrix(self):
        """Return cal_matrix."""
        return self._tens_fitt.cal_matrices[0]

    @cal_matrix.setter
    def cal_matrix(self, new_cal_matrix):
        """set cal_matrix."""
        self._tens_fitt.cal_matrices = [copy.deepcopy(new_cal_matrix)]

    @property
    def state_labels(self):
        """Return state_labels."""
        return self._tens_fitt.substate_labels_list[0]

    @property
    def qubit_list(self):
        """Return list of qubits."""
        return self._qubit_list

    @state_labels.setter
    def state_labels(self, new_state_labels):
        """Set state label."""
        self._tens_fitt.substate_labels_list[0] = new_state_labels

    @property
    def filter(self):
        """Return a measurement filter using the cal matrix."""
        return MeasurementFilter(self.cal_matrix, self.state_labels)

    def add_data(self, new_results, rebuild_cal_matrix=True):
        """
        Add measurement calibration data

        Args:
            new_results (list or qiskit.result.Result): a single result or list
                of result objects.
            rebuild_cal_matrix (bool): rebuild the calibration matrix
        """

        self._tens_fitt.add_data(new_results, rebuild_cal_matrix)

    def subset_fitter(self, qubit_sublist=None):
        """
        Return a fitter object that is a subset of the qubits in the original
        list.

        Args:
            qubit_sublist (list): must be a subset of qubit_list

        Returns:
            CompleteMeasFitter: A new fitter that has the calibration for a
                subset of qubits

        Raises:
            QiskitError: If the calibration matrix is not initialized
        """

        if self._tens_fitt.cal_matrices is None:
            raise QiskitError("Calibration matrix is not initialized")

        if qubit_sublist is None:
            raise QiskitError("Qubit sublist must be specified")

        for qubit in qubit_sublist:
            if qubit not in self._qubit_list:
                raise QiskitError("Qubit not in the original set of qubits")

        # build state labels
        new_state_labels = count_keys(len(qubit_sublist))

        # mapping between indices in the state_labels and the qubits in
        # the sublist
        qubit_sublist_ind = []
        for sqb in qubit_sublist:
            for qbind, qubit in enumerate(self._qubit_list):
                if qubit == sqb:
                    qubit_sublist_ind.append(qbind)

        # states in the full calibration which correspond
        # to the reduced labels
        q_q_mapping = []
        state_labels_reduced = []
        for label in self.state_labels:
            tmplabel = [label[index] for index in qubit_sublist_ind]
            state_labels_reduced.append(''.join(tmplabel))

        for sub_lab_ind, _ in enumerate(new_state_labels):
            q_q_mapping.append([])
            for labelind, label in enumerate(state_labels_reduced):
                if label == new_state_labels[sub_lab_ind]:
                    q_q_mapping[-1].append(labelind)

        new_fitter = CompleteMeasFitter(results=None,
                                        state_labels=new_state_labels,
                                        qubit_list=qubit_sublist)

        new_cal_matrix = np.zeros([len(new_state_labels),
                                   len(new_state_labels)])

        # do a partial trace
        for i in range(len(new_state_labels)):
            for j in range(len(new_state_labels)):

                for q_q_i_map in q_q_mapping[i]:
                    for q_q_j_map in q_q_mapping[j]:
                        new_cal_matrix[i, j] += self.cal_matrix[q_q_i_map,
                                                                q_q_j_map]

                new_cal_matrix[i, j] /= len(q_q_mapping[i])

        new_fitter.cal_matrix = new_cal_matrix

        return new_fitter

    def readout_fidelity(self, label_list=None):
        """
        Based on the results, output the readout fidelity which is the
        normalized trace of the calibration matrix

        Args:
            label_list (bool): If `None`, returns the average assignment fidelity
                of a single state. Otherwise it returns the assignment fidelity
                to be in any one of these states averaged over the second
                index.

        Returns:
            numpy.array: readout fidelity (assignment fidelity)

        Additional Information:
            The on-diagonal elements of the calibration matrix are the
            probabilities of measuring state 'x' given preparation of state
            'x' and so the normalized trace is the average assignment fidelity
        """
        return self._tens_fitt.readout_fidelity(0, label_list)

    def plot_calibration(self, ax=None, show_plot=True):
        """
        Plot the calibration matrix (2D color grid plot)

        Args:
            show_plot (bool): call plt.show()
            ax (matplotlib.axes.Axes): An optional Axes object to use for the
                plot
        """

        self._tens_fitt.plot_calibration(0, ax, show_plot)

class TensoredMeasFitter():
    """
    Measurement correction fitter for a tensored calibration.
    """

    def __init__(self,
                 results: Union[Result, List[Result]],
                 mit_pattern: List[List[int]],
                 substate_labels_list: List[List[str]] = None,
                 circlabel: str = ''):
        """
        Initialize a measurement calibration matrix from the results of running
        the circuits returned by `measurement_calibration_circuits`.

        Args:
            results: the results of running the measurement calibration
                circuits. If this is `None`, the user will set calibration
                matrices later.

            mit_pattern: qubits to perform the
                measurement correction on, divided to groups according to
                tensors

            substate_labels_list: for each
                calibration matrix, the labels of its rows and columns.
                If `None`, the labels are ordered lexicographically

            circlabel: if the qubits were labeled

        Raises:
            ValueError: if the mit_pattern doesn't match the
                substate_labels_list
        """

        self._result_list = []
        self._cal_matrices = None
        self._circlabel = circlabel
        self._mit_pattern = mit_pattern

        self._qubit_list_sizes = \
            [len(qubit_list) for qubit_list in mit_pattern]

        self._indices_list = []
        if substate_labels_list is None:
            self._substate_labels_list = []
            for list_size in self._qubit_list_sizes:
                self._substate_labels_list.append(count_keys(list_size))
        else:
            self._substate_labels_list = substate_labels_list
            if len(self._qubit_list_sizes) != len(substate_labels_list):
                raise ValueError("mit_pattern does not match \
                    substate_labels_list")

        self._indices_list = []
        for _, sub_labels in enumerate(self._substate_labels_list):
            self._indices_list.append(
                {lab: ind for ind, lab in enumerate(sub_labels)})

        self.add_data(results)

    @property
    def cal_matrices(self):
        """Return cal_matrices."""
        return self._cal_matrices

    @cal_matrices.setter
    def cal_matrices(self, new_cal_matrices):
        """Set _cal_matrices."""
        self._cal_matrices = copy.deepcopy(new_cal_matrices)

    @property
    def substate_labels_list(self):
        """Return _substate_labels_list."""
        return self._substate_labels_list

    @property
    def filter(self):
        """Return a measurement filter using the cal matrices."""
        return TensoredFilter(self._cal_matrices, self._substate_labels_list, self._mit_pattern)

    @property
    def nqubits(self):
        """Return _qubit_list_sizes."""
        return sum(self._qubit_list_sizes)

    def add_data(self, new_results, rebuild_cal_matrix=True):
        """
        Add measurement calibration data

        Args:
            new_results (list or qiskit.result.Result): a single result or list
                of Result objects.
            rebuild_cal_matrix (bool): rebuild the calibration matrix
        """

        if new_results is None:
            return

        if not isinstance(new_results, list):
            new_results = [new_results]

        for result in new_results:
            self._result_list.append(result)

        if rebuild_cal_matrix:
            self._build_calibration_matrices()

    def readout_fidelity(self, cal_index=0, label_list=None):
        """
        Based on the results, output the readout fidelity, which is the average
        of the diagonal entries in the calibration matrices.

        Args:
            cal_index(integer): readout fidelity for this index in _cal_matrices
            label_list (list):  Returns the average fidelity over of the groups
                f states. In the form of a list of lists of states. If `None`,
                then each state used in the construction of the calibration
                matrices forms a group of size 1

        Returns:
            numpy.array: The readout fidelity (assignment fidelity)

        Raises:
            QiskitError: If the calibration matrix has not been set for the
                object.

        Additional Information:
            The on-diagonal elements of the calibration matrices are the
            probabilities of measuring state 'x' given preparation of state
            'x'.
        """

        if self._cal_matrices is None:
            raise QiskitError("Cal matrix has not been set")

        if label_list is None:
            label_list = [[label] for label in
                          self._substate_labels_list[cal_index]]

        state_labels = self._substate_labels_list[cal_index]
        fidelity_label_list = []
        if label_list is None:
            fidelity_label_list = [[label] for label in state_labels]
        else:
            for fid_sublist in label_list:
                fidelity_label_list.append([])
                for fid_statelabl in fid_sublist:
                    for label_idx, label in enumerate(state_labels):
                        if fid_statelabl == label:
                            fidelity_label_list[-1].append(label_idx)
                            continue

        # fidelity_label_list is a 2D list of indices in the
        # cal_matrix, we find the assignment fidelity of each
        # row and average over the list
        assign_fid_list = []

        for fid_label_sublist in fidelity_label_list:
            assign_fid_list.append(0)
            for state_idx_i in fid_label_sublist:
                for state_idx_j in fid_label_sublist:
                    assign_fid_list[-1] += \
                        self._cal_matrices[cal_index][state_idx_i][state_idx_j]
            assign_fid_list[-1] /= len(fid_label_sublist)

        return np.mean(assign_fid_list)

    def _build_calibration_matrices(self):
        """
        Build the measurement calibration matrices from the results of running
        the circuits returned by `measurement_calibration`.
        """

        # initialize the set of empty calibration matrices
        self._cal_matrices = []
        for list_size in self._qubit_list_sizes:
            self._cal_matrices.append(np.zeros([2**list_size, 2**list_size],
                                               dtype=float))

        # go through for each calibration experiment
        for result in self._result_list:
            for experiment in result.results:
                circ_name = experiment.header.name
                # extract the state from the circuit name
                # this was the prepared state
                circ_search = re.search('(?<=' + self._circlabel + 'cal_)\\w+',
                                        circ_name)

                # this experiment is not one of the calcs so skip
                if circ_search is None:
                    continue

                state = circ_search.group(0)

                # get the counts from the result
                state_cnts = result.get_counts(circ_name)
                for measured_state, counts in state_cnts.items():
                    end_index = self.nqubits
                    for cal_ind, cal_mat in enumerate(self._cal_matrices):

                        start_index = end_index - \
                            self._qubit_list_sizes[cal_ind]

                        substate_index = self._indices_list[cal_ind][
                            state[start_index:end_index]]
                        measured_substate_index = \
                            self._indices_list[cal_ind][
                                measured_state[start_index:end_index]]
                        end_index = start_index

                        cal_mat[measured_substate_index][substate_index] += \
                            counts

        for mat_index, _ in enumerate(self._cal_matrices):
            sums_of_columns = np.sum(self._cal_matrices[mat_index], axis=0)
            # pylint: disable=assignment-from-no-return
            self._cal_matrices[mat_index] = np.divide(
                self._cal_matrices[mat_index], sums_of_columns,
                out=np.zeros_like(self._cal_matrices[mat_index]),
                where=sums_of_columns != 0)

    def plot_calibration(self, cal_index=0, ax=None, show_plot=True):
        """
        Plot one of the calibration matrices (2D color grid plot).

        Args:
            cal_index(integer): calibration matrix to plot
            ax(matplotlib.axes): settings for the graph
            show_plot (bool): call plt.show()

        Raises:
            QiskitError: if _cal_matrices was not set.

            ImportError: if matplotlib was not installed.

        """

        if self._cal_matrices is None:
            raise QiskitError("Cal matrix has not been set")

        if not HAS_MATPLOTLIB:
            raise ImportError('The function plot_rb_data needs matplotlib. '
                              'Run "pip install matplotlib" before.')

        if ax is None:
            plt.figure()
            ax = plt.gca()

        axim = ax.matshow(self.cal_matrices[cal_index],
                          cmap=plt.cm.binary,
                          clim=[0, 1])
        ax.figure.colorbar(axim)
        ax.set_xlabel('Prepared State')
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('Measured State')
        ax.set_xticks(np.arange(len(self._substate_labels_list[cal_index])))
        ax.set_yticks(np.arange(len(self._substate_labels_list[cal_index])))
        ax.set_xticklabels(self._substate_labels_list[cal_index])
        ax.set_yticklabels(self._substate_labels_list[cal_index])

        if show_plot:
            plt.show()














class MeasurementFilter():
    """
    Measurement error mitigation filter.

    Produced from a measurement calibration fitter and can be applied
    to data.

    """

    def __init__(self,
                 cal_matrix: np.matrix,
                 state_labels: list):
        """
        Initialize a measurement error mitigation filter using the cal_matrix
        from a measurement calibration fitter.

        Args:
            cal_matrix: the calibration matrix for applying the correction
            state_labels: the states for the ordering of the cal matrix
        """

        self._cal_matrix = cal_matrix
        self._state_labels = state_labels

    @property
    def cal_matrix(self):
        """Return cal_matrix."""
        return self._cal_matrix

    @property
    def state_labels(self):
        """return the state label ordering of the cal matrix"""
        return self._state_labels

    @state_labels.setter
    def state_labels(self, new_state_labels):
        """set the state label ordering of the cal matrix"""
        self._state_labels = new_state_labels

    @cal_matrix.setter
    def cal_matrix(self, new_cal_matrix):
        """Set cal_matrix."""
        self._cal_matrix = new_cal_matrix

    def apply(self,
              raw_data,
              method='least_squares'):
        """Apply the calibration matrix to results.

        Args:
            raw_data (dict or list): The data to be corrected. Can be in a number of forms:

                 Form 1: a counts dictionary from results.get_counts

                 Form 2: a list of counts of `length==len(state_labels)`

                 Form 3: a list of counts of `length==M*len(state_labels)` where M is an
                 integer (e.g. for use with the tomography data)

                 Form 4: a qiskit Result

            method (str): fitting method. If `None`, then least_squares is used.

                ``pseudo_inverse``: direct inversion of the A matrix

                ``least_squares``: constrained to have physical probabilities

        Returns:
            dict or list: The corrected data in the same form as `raw_data`

        Raises:
            QiskitError: if `raw_data` is not an integer multiple
                of the number of calibrated states.

        """

        # check forms of raw_data
        if isinstance(raw_data, dict):
            # counts dictionary
            for data_label in raw_data.keys():
                if data_label not in self._state_labels:
                    raise QiskitError("Unexpected state label '" + data_label +
                                      "', verify the fitter's state labels "
                                      "correspond to the input data")
            data_format = 0
            # convert to form2
            raw_data2 = [np.zeros(len(self._state_labels), dtype=float)]
            for stateidx, state in enumerate(self._state_labels):
                raw_data2[0][stateidx] = raw_data.get(state, 0)

        elif isinstance(raw_data, list):
            size_ratio = len(raw_data)/len(self._state_labels)
            if len(raw_data) == len(self._state_labels):
                data_format = 1
                raw_data2 = [raw_data]
            elif int(size_ratio) == size_ratio:
                data_format = 2
                size_ratio = int(size_ratio)
                # make the list into chunks the size of state_labels for easier
                # processing
                raw_data2 = np.zeros([size_ratio, len(self._state_labels)])
                for i in range(size_ratio):
                    raw_data2[i][:] = raw_data[
                        i * len(self._state_labels):(i + 1)*len(
                            self._state_labels)]
            else:
                raise QiskitError("Data list is not an integer multiple "
                                  "of the number of calibrated states")

        elif isinstance(raw_data, qiskit.result.result.Result):

            # extract out all the counts, re-call the function with the
            # counts and push back into the new result
            new_result = deepcopy(raw_data)

            new_counts_list = parallel_map(
                self._apply_correction,
                [resultidx for resultidx, _ in enumerate(raw_data.results)],
                task_args=(raw_data, method))

            for resultidx, new_counts in new_counts_list:
                new_result.results[resultidx].data.counts = new_counts

            return new_result

        else:
            raise QiskitError("Unrecognized type for raw_data.")

        if method == 'pseudo_inverse':
            pinv_cal_mat = la.pinv(self._cal_matrix)

        # Apply the correction
        for data_idx, _ in enumerate(raw_data2):

            if method == 'pseudo_inverse':
                raw_data2[data_idx] = np.dot(
                    pinv_cal_mat, raw_data2[data_idx])

            elif method == 'least_squares':
                nshots = sum(raw_data2[data_idx])

                def fun(x):
                    return sum(
                        (raw_data2[data_idx] - np.dot(self._cal_matrix, x))**2)
                x0 = np.random.rand(len(self._state_labels))
                x0 = x0 / sum(x0)
                cons = ({'type': 'eq', 'fun': lambda x: nshots - sum(x)})
                bnds = tuple((0, nshots) for x in x0)
                res = minimize(fun, x0, method='SLSQP',
                               constraints=cons, bounds=bnds, tol=1e-6)
                raw_data2[data_idx] = res.x

            else:
                raise QiskitError("Unrecognized method.")

        if data_format == 2:
            # flatten back out the list
            raw_data2 = raw_data2.flatten()

        elif data_format == 0:
            # convert back into a counts dictionary
            new_count_dict = {}
            for stateidx, state in enumerate(self._state_labels):
                if raw_data2[0][stateidx] != 0:
                    new_count_dict[state] = raw_data2[0][stateidx]

            raw_data2 = new_count_dict
        else:
            # TODO: should probably change to:
            # raw_data2 = raw_data2[0].tolist()
            raw_data2 = raw_data2[0]
        return raw_data2

    def _apply_correction(self, resultidx, raw_data, method):
        """Wrapper to call apply with a counts dictionary."""
        new_counts = self.apply(
            raw_data.get_counts(resultidx), method=method)
        return resultidx, new_counts
    

class TensoredFilter():
    """
    Tensored measurement error mitigation filter.

    Produced from a tensored measurement calibration fitter and can be applied
    to data.
    """

    def __init__(self,
                 cal_matrices: np.matrix,
                 substate_labels_list: list,
                 mit_pattern: list):
        """
        Initialize a tensored measurement error mitigation filter using
        the cal_matrices from a tensored measurement calibration fitter.
        A simple usage this class is explained [here]
        (https://qiskit.org/documentation/tutorials/noise/3_measurement_error_mitigation.html).

        Args:
            cal_matrices: the calibration matrices for applying the correction.
            substate_labels_list: for each calibration matrix
                a list of the states (as strings, states in the subspace)
            mit_pattern: for each calibration matrix
                a list of the logical qubit indices (as int, states in the subspace)
        """

        self._cal_matrices = cal_matrices
        self._qubit_list_sizes = []
        self._indices_list = []
        self._substate_labels_list = []
        self.substate_labels_list = substate_labels_list
        self._mit_pattern = mit_pattern

    @property
    def cal_matrices(self):
        """Return cal_matrices."""
        return self._cal_matrices

    @cal_matrices.setter
    def cal_matrices(self, new_cal_matrices):
        """Set cal_matrices."""
        self._cal_matrices = deepcopy(new_cal_matrices)

    @property
    def substate_labels_list(self):
        """Return _substate_labels_list"""
        return self._substate_labels_list

    @substate_labels_list.setter
    def substate_labels_list(self, new_substate_labels_list):
        """Return _substate_labels_list"""
        self._substate_labels_list = new_substate_labels_list

        # get the number of qubits in each subspace
        self._qubit_list_sizes = []
        for _, substate_label_list in enumerate(self._substate_labels_list):
            self._qubit_list_sizes.append(
                int(np.log2(len(substate_label_list))))

        # get the indices in the calibration matrix
        self._indices_list = []
        for _, sub_labels in enumerate(self._substate_labels_list):

            self._indices_list.append(
                {lab: ind for ind, lab in enumerate(sub_labels)})

    @property
    def qubit_list_sizes(self):
        """Return _qubit_list_sizes."""
        return self._qubit_list_sizes

    @property
    def nqubits(self):
        """Return the number of qubits. See also MeasurementFilter.apply() """
        return sum(self._qubit_list_sizes)

    def apply(self,
              raw_data: Union[qiskit.result.result.Result, dict],
              method: str = 'least_squares',
              meas_layout: List[int] = None):
        """
        Apply the calibration matrices to results.

        Args:
            raw_data (dict or Result): The data to be corrected. Can be in one of two forms:

                * A counts dictionary from results.get_counts

                * A Qiskit Result

            method (str): fitting method. The following methods are supported:

                * 'pseudo_inverse': direct inversion of the cal matrices.
                    Mitigated counts can contain negative values
                    and the sum of counts would not equal to the shots.
                    Mitigation is conducted qubit wise:
                    For each qubit, mitigate the whole counts using the calibration matrices
                    which affect the corresponding qubit.
                    For example, assume we are mitigating the 3rd bit of the 4-bit counts
                    using '2\times 2' calibration matrix `A_3`.
                    When mitigating the count of '0110' in this step,
                    the following formula is applied:
                    `count['0110'] = A_3^{-1}[1, 0]*count['0100'] + A_3^{-1}[1, 1]*count['0110']`.

                    The total time complexity of this method is `O(m2^{n + t})`,
                    where `n` is the size of calibrated qubits,
                    `m` is the number of sets in `mit_pattern`,
                    and `t` is the size of largest set of mit_pattern.
                    If the `mit_pattern` is shaped like `[[0], [1], [2], ..., [n-1]]`,
                    which corresponds to the tensor product noise model without cross-talk,
                    then the time complexity would be `O(n2^n)`.
                    If the `mit_pattern` is shaped like `[[0, 1, 2, ..., n-1]]`,
                    which exactly corresponds to the complete error mitigation,
                    then the time complexity would be `O(2^(n+n)) = O(4^n)`.


                * 'least_squares': constrained to have physical probabilities.
                    Instead of directly applying inverse calibration matrices,
                    this method solve a constrained optimization problem to find
                    the closest probability vector to the result from 'pseudo_inverse' method.
                    Sequential least square quadratic programming (SLSQP) is used
                    in the internal process.
                    Every updating step in SLSQP takes `O(m2^{n+t})` time.
                    Since this method is using the SLSQP optimization over
                    the vector with lenght `2^n`, the mitigation for 8 bit counts
                    with the `mit_pattern = [[0], [1], [2], ..., [n-1]]` would
                    take 10 seconds or more.

                * If `None`, 'least_squares' is used.

            meas_layout (list of int): the mapping from classical registers to qubits

                * If you measure qubit `2` to clbit `0`, `0` to `1`, and `1` to `2`,
                    the list becomes `[2, 0, 1]`

                * If `None`, flatten(mit_pattern) is used.

        Returns:
            dict or Result: The corrected data in the same form as raw_data

        Raises:
            QiskitError: if raw_data is not in a one of the defined forms.
        """

        all_states = count_keys(self.nqubits)
        num_of_states = 2**self.nqubits

        if meas_layout is None:
            meas_layout = []
            for qubits in self._mit_pattern:
                meas_layout += qubits

        # check forms of raw_data
        if isinstance(raw_data, dict):
            # counts dictionary
            # convert to list
            raw_data2 = [np.zeros(num_of_states, dtype=float)]
            for state, count in raw_data.items():
                stateidx = int(state, 2)
                raw_data2[0][stateidx] = count

        elif isinstance(raw_data, qiskit.result.result.Result):

            # extract out all the counts, re-call the function with the
            # counts and push back into the new result
            new_result = deepcopy(raw_data)

            new_counts_list = parallel_map(
                self._apply_correction,
                [resultidx for resultidx, _ in enumerate(raw_data.results)],
                task_args=(raw_data, method, meas_layout))

            for resultidx, new_counts in new_counts_list:
                new_result.results[resultidx].data.counts = new_counts

            return new_result

        else:
            raise QiskitError("Unrecognized type for raw_data.")

        if method == 'pseudo_inverse':
            pinv_cal_matrices = []
            for cal_mat in self._cal_matrices:
                pinv_cal_matrices.append(la.pinv(cal_mat))

        meas_layout = meas_layout[::-1]  # reverse endian
        qubits_to_clbits = [-1 for _ in range(max(meas_layout) + 1)]
        for i, qubit in enumerate(meas_layout):
            qubits_to_clbits[qubit] = i

        # Apply the correction
        for data_idx, _ in enumerate(raw_data2):

            if method == 'pseudo_inverse':
                for pinv_cal_mat, pos_qubits, indices in zip(pinv_cal_matrices,
                                                             self._mit_pattern,
                                                             self._indices_list):
                    inv_mat_dot_x = np.zeros([num_of_states], dtype=float)
                    pos_clbits = [qubits_to_clbits[qubit] for qubit in pos_qubits]
                    for state_idx, state in enumerate(all_states):
                        first_index = self.compute_index_of_cal_mat(state, pos_clbits, indices)
                        for i in range(len(pinv_cal_mat)):  # i is index of pinv_cal_mat
                            source_state = self.flip_state(state, i, pos_clbits)
                            second_index = self.compute_index_of_cal_mat(source_state,
                                                                         pos_clbits,
                                                                         indices)
                            inv_mat_dot_x[state_idx] += pinv_cal_mat[first_index, second_index]\
                                * raw_data2[data_idx][int(source_state, 2)]
                    raw_data2[data_idx] = inv_mat_dot_x

            elif method == 'least_squares':
                def fun(x):
                    mat_dot_x = deepcopy(x)
                    for cal_mat, pos_qubits, indices in zip(self._cal_matrices,
                                                            self._mit_pattern,
                                                            self._indices_list):
                        res_mat_dot_x = np.zeros([num_of_states], dtype=float)
                        pos_clbits = [qubits_to_clbits[qubit] for qubit in pos_qubits]
                        for state_idx, state in enumerate(all_states):
                            second_index = self.compute_index_of_cal_mat(state, pos_clbits, indices)
                            for i in range(len(cal_mat)):
                                target_state = self.flip_state(state, i, pos_clbits)
                                first_index =\
                                    self.compute_index_of_cal_mat(target_state, pos_clbits, indices)
                                res_mat_dot_x[int(target_state, 2)]\
                                    += cal_mat[first_index, second_index] * mat_dot_x[state_idx]
                        mat_dot_x = res_mat_dot_x
                    return sum((raw_data2[data_idx] - mat_dot_x) ** 2)

                x0 = np.random.rand(num_of_states)
                x0 = x0 / sum(x0)
                nshots = sum(raw_data2[data_idx])
                cons = ({'type': 'eq', 'fun': lambda x: nshots - sum(x)})
                bnds = tuple((0, nshots) for x in x0)
                res = minimize(fun, x0, method='SLSQP',
                               constraints=cons, bounds=bnds, tol=1e-6)
                raw_data2[data_idx] = res.x

            else:
                raise QiskitError("Unrecognized method.")

        # convert back into a counts dictionary
        new_count_dict = {}
        for state_idx, state in enumerate(all_states):
            if raw_data2[0][state_idx] != 0:
                new_count_dict[state] = raw_data2[0][state_idx]

        return new_count_dict

    def flip_state(self, state: str, mat_index: int, flip_poses: List[int]) -> str:
        """Flip the state according to the chosen qubit positions"""
        flip_poses = [pos for i, pos in enumerate(flip_poses) if (mat_index >> i) & 1]
        flip_poses = sorted(flip_poses)
        new_state = ""
        pos = 0
        for flip_pos in flip_poses:
            new_state += state[pos:flip_pos]
            new_state += str(int(state[flip_pos], 2) ^ 1)  # flip the state
            pos = flip_pos + 1
        new_state += state[pos:]
        return new_state

    def compute_index_of_cal_mat(self, state: str, pos_qubits: List[int], indices: dict) -> int:
        """Return the index of (pseudo inverse) calibration matrix for the input quantum state"""
        sub_state = ""
        for pos in pos_qubits:
            sub_state += state[pos]
        return indices[sub_state]

    def _apply_correction(self,
                          resultidx: int,
                          raw_data: qiskit.result.result.Result,
                          method: str,
                          meas_layout: List[int]):
        """Wrapper to call apply with a counts dictionary."""
        new_counts = self.apply(
            raw_data.get_counts(resultidx), method=method, meas_layout=meas_layout)
        return resultidx, new_counts
    




