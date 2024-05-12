from collections import Counter
import outlines
from evaluate import load

class ExperimentModule:
    def __init__(self, data_module, model_module):
        self.data_module = data_module
        self.model_module = model_module
        self.model = self.model_module.load_outlines_model()

    def run_experiment(self, prompt, sampling_params, exp=None):
        bleu = load("bleu")
        results = {}  # results 딕셔너리 초기화
        
        if exp == "cot" or exp == "sc":
            # Chain-of-Thought & Self-Consistency Voting
            questions = self.data_module.generate_questions(prompt, exp)
            answers = self.model_module.generate_answers(questions, sampling_params)
            generator = outlines.generate.choice(self.model, ['A', 'B'])
            choice_questions = self.data_module.prepare_for_choice(prompt, answers)
            final_answers = generator(choice_questions)
            Gen_answers = [answer.split("Response:")[-1].strip() for answer in answers]  
            
            bleu_scores = []
            for gen_answer, final_answer, row in zip(Gen_answers, final_answers, self.data_module.data_frame.itertuples(index=False)):
                if final_answer == 'A':
                    reference = [row.us]
                else:
                    reference = [row.ko]
                bleu_score = bleu.compute(predictions=[gen_answer], references=reference)['bleu']
                bleu_scores.append(bleu_score)

            if exp == 'sc':
                final_answers = [Counter(final_answers[i:i+3]).most_common()[0][0] for i in range(0, len(final_answers), 3)]
            results.update(self.count_answers(final_answers))  # results에 US, KO 개수 추가
            results['cot'] = answers
            results['generated_answers'] = final_answers
            results['questions'] = questions  
            results['model_answer'] = Gen_answers
            results['bleu_scores'] = bleu_scores

        else:
            # multiple choice
            questions = self.data_module.generate_questions(prompt, exp)
            answers = self.model_module.generate_answers(questions, sampling_params)
            generator = outlines.generate.choice(self.model, ['A', 'B'])
            final_answers = generator(questions)
            Gen_answers = [answer.split("Response:")[-1].strip() for answer in answers]  # 실제 텍스트 답변 추출
            
            bleu_scores = []
            for gen_answer, final_answer, row in zip(Gen_answers, final_answers, self.data_module.data_frame.itertuples(index=False)):
                if final_answer == 'A':
                    reference = [row.us]
                else:
                    reference = [row.ko]
                bleu_score = bleu.compute(predictions=[gen_answer], references=reference)['bleu']
                bleu_scores.append(bleu_score)

            results.update(self.count_answers(final_answers))  # results에 US, KO 개수 추가
            results['generated_answers'] = final_answers
            results['questions'] = questions  # 질문 저장
            results['model_answer'] = Gen_answers
            results['bleu_scores'] = bleu_scores

        return results

    @staticmethod
    def count_answers(answers):
        """Count the frequency of answers and remap them for clarity!"""
        counts = dict(Counter(answers))
        return {'US': counts.get('A', 0), 'KO': counts.get('B', 0)}
