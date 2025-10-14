#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic conversion of STL containers
#include <math.h>
#include <float.h> // For DBL_MAX
#include <vector>
#include <string>
#include <map>

namespace py = pybind11;

int _calc(py::dict FREQ, py::object sentence, py::dict DAG, py::dict route, double total)
{
    const py::ssize_t N = py::len(sentence);
    const double logtotal = log(total);
    double max_freq, fq_last; // Declared fq_last here
    py::ssize_t max_x, idx, i, t_list_len, x;

    route[py::int_(N)] = py::make_tuple(py::int_(0), py::int_(0));

    for(idx = N - 1; idx >= 0 ;idx--)
    {
        max_freq = -DBL_MAX; // Equivalent to INT_MIN for double
        max_x = 0;
        py::list t_list = DAG[py::int_(idx)];
        t_list_len = py::len(t_list);
        for(i = 0; i < t_list_len; i++)
        {
            double fq = 1;
            x = py::cast<py::ssize_t>(t_list[i]);
            py::object slice_of_sentence = sentence.attr("__getitem__")(py::slice(idx, x + 1, 1));
            py::object o_freq = FREQ.attr("get")(slice_of_sentence); // Use .get() to avoid KeyError

            if (!o_freq.is_none()) // Check if o_freq is not NULL
            {
                fq = py::cast<double>(o_freq);
                if (fq == 0) fq = 1;
            }
            py::tuple t_tuple = route[py::int_(x + 1)];
            double fq_2 = py::cast<double>(t_tuple[0]);
            fq_last = log(fq) - logtotal + fq_2;
            if(fq_last > max_freq)
            {
                max_freq = fq_last;
                max_x = x;
            }
        }
        route[py::int_(idx)] = py::make_tuple(max_freq, py::int_(max_x));
    }
    return 1;
}

int _get_DAG(py::dict DAG, py::dict FREQ, py::object sentence)
{
    const py::ssize_t N = py::len(sentence);
    py::object frag;
    py::ssize_t i, k;
    for(k = 0; k < N; k++)
    {
        py::list tmplist; // Equivalent to PyList_New(0)
        i = k;
        frag = sentence.attr("__getitem__")(k); // Explicitly get item as py::object
        while(i < N && FREQ.contains(frag)) // Equivalent to PyDict_Contains(FREQ, frag)
        {
            if (py::cast<py::ssize_t>(FREQ[frag])) // Check if frequency is non-zero
            {
                tmplist.append(py::int_(i)); // Equivalent to PyList_Append(tmplist, PyInt_FromLong((long)i))
            }
            i += 1;
            frag = sentence.attr("__getitem__")(py::slice(k, i + 1, 1)); // Explicitly get slice as py::object
        }
        if (py::len(tmplist) == 0) // Equivalent to PyList_Size(tmplist) == 0
            tmplist.append(py::int_(k)); // Equivalent to PyList_Append(tmplist, PyInt_FromLong((long)k))
        DAG[py::int_(k)] = tmplist; // Equivalent to PyDict_SetItem(DAG, PyInt_FromLong((long)k), tmplist)
    }
    return 1;
}

int _get_DAG_and_calc(py::dict FREQ, py::object sentence, py::list route, double total)
{
    const py::ssize_t N = py::len(sentence);
    std::vector<std::vector<py::ssize_t>> DAG(N, std::vector<py::ssize_t>(20));
    std::vector<py::ssize_t> points(N, 0);
    std::vector<std::vector<double>> _route(N + 1, std::vector<double>(2));

    double logtotal = log(total);
    double max_freq_val;
    double fq_2, fq_last;

    _route[N][0] = 0;
    _route[N][1] = 0;

    for(py::ssize_t k = 0; k < N; k++)
    {
        py::ssize_t i = k;
        py::object frag = sentence.attr("__getitem__")(k); // Explicitly get item as py::object
        py::object t_f;

        while(i < N && (t_f = FREQ.attr("get")(frag)) && (points[k] < 12))
        {
            if (py::cast<py::ssize_t>(t_f))
            {
                DAG[k][points[k]] = i;
                points[k]++;
            }
            i++;
            frag = sentence.attr("__getitem__")(py::slice(k, i + 1, 1)); // Explicitly get slice as py::object
        }
        if (points[k] == 0)
        {
            DAG[k][0] = k;
            points[k] = 1;
        }
    }

    for(py::ssize_t idx = N - 1; idx >= 0 ;idx--)
    {
        max_freq_val = -DBL_MAX;
        py::ssize_t max_x = 0;
        py::ssize_t t_list_len = points[idx];
        for(py::ssize_t i = 0; i < t_list_len; i++)
        {
            double fq = 1;
            py::ssize_t x = DAG[idx][i];
            py::object slice_of_sentence = sentence.attr("__getitem__")(py::slice(idx, x + 1, 1));
            py::object o_freq = FREQ.attr("get")(slice_of_sentence);

            if (!o_freq.is_none())
            {
                fq = py::cast<double>(o_freq);
                if (fq == 0) fq = 1;
            }
            fq_2 = _route[x + 1][0];
            fq_last = log(fq) - logtotal + fq_2;
            if(fq_last >= max_freq_val)
            {
                max_freq_val = fq_last;
                max_x = x;
            }
        }
        _route[idx][0] = max_freq_val;
        _route[idx][1] = (double)max_x;
    }
    for(py::ssize_t i = 0; i <= N; i++)
    {
        route.append(py::int_((long)_route[i][1]));
    }
    return 1;
}

py::tuple _viterbi(py::object obs, py::str _states, py::dict start_p, py::dict trans_p, py::dict emip_p)
{
    const py::ssize_t obs_len = py::len(obs);
    const int states_num = 4;

    std::vector<std::vector<double>> V(obs_len, std::vector<double>(22));
    std::vector<std::vector<char>> path(obs_len, std::vector<char>(22));

    std::string states_str = _states;
    const char* states = states_str.c_str();

    char y, best_state, y0, now_state;
    int p;

    std::map<char, std::string> PrevStatus_map;
    PrevStatus_map['B'] = "ES";
    PrevStatus_map['M'] = "MB";
    PrevStatus_map['S'] = "SE";
    PrevStatus_map['E'] = "BM";

    std::vector<py::object> emip_p_dict(states_num);
    std::vector<std::vector<py::object>> trans_p_dict(22, std::vector<py::object>(2));
    std::vector<py::object> py_states(states_num);

    for(int i = 0; i < states_num; i++)
        py_states[i] = py::str(std::string(1, states[i])); // Fix: Convert char to std::string

    emip_p_dict[0] = emip_p[py_states[0]];
    emip_p_dict[1] = emip_p[py_states[1]];
    emip_p_dict[2] = emip_p[py_states[2]];
    emip_p_dict[3] = emip_p[py_states[3]];

    trans_p_dict['B' - 'B'][0] = trans_p[py_states[2]];
    trans_p_dict['B' - 'B'][1] = trans_p[py_states[3]];
    trans_p_dict['M' - 'B'][0] = trans_p[py_states[1]];
    trans_p_dict['M' - 'B'][1] = trans_p[py_states[0]];
    trans_p_dict['E' - 'B'][0] = trans_p[py_states[0]];
    trans_p_dict['E' - 'B'][1] = trans_p[py_states[1]];
    trans_p_dict['S' - 'B'][0] = trans_p[py_states[3]];
    trans_p_dict['S' - 'B'][1] = trans_p[py_states[2]];

    for(int i = 0; i < states_num; i++)
    {
        py::object t_dict = emip_p[py_states[i]];
        double t_double = -DBL_MAX;
        py::object ttemp = obs.attr("__getitem__")(0); // Explicitly get item as py::object
        py::object item = t_dict.attr("get")(ttemp);
        if(!item.is_none())
            t_double = py::cast<double>(item);
        double t_double_2 = py::cast<double>(start_p[py_states[i]]);
        V[0][states[i] - 'B'] = t_double + t_double_2;
        path[0][states[i] - 'B'] = states[i];
    }

    for(py::ssize_t i = 1; i < obs_len; i++)
    {
        py::object t_obs = obs.attr("__getitem__")(i); // Explicitly get item as py::object
        for(int j = 0; j < states_num; j++)
        {
            double em_p = -DBL_MAX;
            y = states[j];
            py::object item = emip_p_dict[j].attr("get")(t_obs);
            if(!item.is_none())
                em_p = py::cast<double>(item);
            double max_prob = -DBL_MAX;
            best_state = '\0';
            for(p = 0; p < 2; p++)
            {
                double prob = em_p;
                y0 = PrevStatus_map[y][p];
                prob += V[i - 1][y0 - 'B'];
                item = trans_p_dict[y - 'B'][p].attr("get")(py_states[j]);
                if (item.is_none())
                    prob += -DBL_MAX;
                else
                    prob += py::cast<double>(item);
                if (prob > max_prob)
                {
                    max_prob = prob;
                    best_state = y0;
                }
            }
            if(best_state == '\0')
            {
                for(p = 0; p < 2; p++)
                {
                    y0 = PrevStatus_map[y][p];
                    if(y0 > best_state)
                        best_state = y0;
                }
            }
            V[i][y - 'B'] = max_prob;
            path[i][y - 'B'] = best_state;
        }
    }

    double max_prob_final = V[obs_len - 1]['E' - 'B'];
    best_state = 'E';

    if (V[obs_len - 1]['S' - 'B'] > max_prob_final)
    {
        max_prob_final = V[obs_len - 1]['S' - 'B'];
        best_state = 'S';
    }

    py::list t_list;
    now_state = best_state;

    for(py::ssize_t i = obs_len - 1; i >= 0; i--)
    {
        t_list.insert(0, py::str(std::string(1, now_state))); // Fix: Convert char to std::string
        now_state = path[i][now_state - 'B'];
    }

    return py::make_tuple(max_prob_final, t_list);
}

PYBIND11_MODULE(_jieba_fast_functions_pybind, m) {
    m.doc() = "pybind11 plugin for jieba_fast_dat"; // optional module docstring

    m.def("_calc", &_calc, "A function to calculate word frequencies");
    m.def("_get_DAG", &_get_DAG, "A function to get DAG");
    m.def("_get_DAG_and_calc", &_get_DAG_and_calc, "A function to get DAG and calculate");
    m.def("_viterbi", &_viterbi, "A function to perform Viterbi algorithm");
}