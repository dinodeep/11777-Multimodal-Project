import re
import nltk

def remove_punc_split_by_word(s):
    """Removes punctutation from a string and splits it by word

    Args:
        story (str): string to remove punctuation and split by word

    Returns:
        _type_: _description_
    """
    s = re.sub(r'[^a-zA-Z0-9_ ]+', '', s)
    s = s.split(" ")
    return s  

def inter_story_trigram_repetition(predicted_story):
    """ Calculates (total # trigrams - # unique trigrams) / (total # trigrams)
    
    Args:
        predicted_story: list(str) -- list of tokens representing strings    

    Returns:
        float: float between 0 and 1 representing reptition
    """
    trigrams = {}
    # story_by_word = remove_punc_split_by_word(predicted_story)
    print(predicted_story)
    total_trigrams = 0
    for idx in range(0, len(predicted_story)-2):
        seq = predicted_story[idx:idx+3]
        trigram = " ".join(seq)
        if trigram in trigrams:
          trigrams[trigram] += 1
        else:
          trigrams[trigram] = 1
        total_trigrams += 1

    num_unique_trigrams = 0
    for trigram in trigrams:
        if trigrams[trigram] == 1:
          num_unique_trigrams += 1

    return (total_trigrams - num_unique_trigrams) / total_trigrams


def calculate_metrics(data, file_path):
    results = []
    meteor_scores = 0.0
    bleu_scores = 0.0
    fluency_scores = 0.0

    num_stories = 0
    for image_seq, pred_story in data:
        # print("Image Seq: ", image_seq)
        # print("Pred Story", pred_story)
        gt_story = []
        printing = []
        pred_story = pred_story[0]
        for img, img_label in image_seq:
            # print(img_label)
            img_label = re.sub(r'[^a-zA-Z0-9_ ]+', '', img_label)
            # print(img_label)
            gt_story.append(img_label.strip().split(" "))
            printing.append(img_label)

        # Remove punctuation and split into a list of words

        pred_story = remove_punc_split_by_word(pred_story)
        # gt_story = remove_punc_split_by_word(gt_story)

        # Calculate meteor score
        meteor_score = -1
        for sentence in gt_story:
            tmp_meteor_score = nltk.translate.meteor_score.single_meteor_score(sentence, pred_story)
            # print(tmp_meteor_score)
            if tmp_meteor_score > meteor_score:
              meteor_score = tmp_meteor_score

        meteor_scores += meteor_score

        bleu_score = -1
        for sentence in gt_story:
            tmp_bleu_score = nltk.translate.bleu_score.sentence_bleu(sentence, pred_story)
            if tmp_bleu_score > bleu_score:
              bleu_score = tmp_bleu_score
        bleu_scores += bleu_score

        fluency = inter_story_trigram_repetition(pred_story)
        fluency_scores += fluency

        print(f"Meteor: {meteor_score}, Bleu: {bleu_score}, Fluency: {fluency}")
        results.append([pred_story, gt_story, meteor_score, bleu_score, fluency])
        num_stories += 1


    avg_meteor = meteor_scores / num_stories
    avg_bleu = bleu_scores / num_stories
    avg_fluency = fluency_scores / num_stories
    print(f"Avg Meteor: {avg_meteor}, Avg Bleu: {avg_bleu}, Avg Fluency: {avg_fluency}")

    #with open(f'{file_path}_metrics.json', 'w') as f:
        #json.dump(results, f)