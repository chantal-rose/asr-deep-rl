import Levenshtein

def calculate_levenshtein(h, y, lh, ly, decoder, labels):

    beam_results, beam_scores, timesteps, out_lens = decoder.decode(h, seq_lens = lh)

    batch_size = out_lens.shape[0]
    distance = 0

    for i in range(batch_size):
      h_sliced = beam_results[i][0][:out_lens[i][0]]
      y_sliced = y[i][:ly[i]]
      h_string = "".join([labels[n] for n in h_sliced])
      y_string = "".join([labels[n] for n in y_sliced])
      distance += Levenshtein.distance(h_string, y_string)
      # /len(y_string)

    distance /= batch_size

    return distance