"""
Test Suite for Professional Hybrid Detector
Tests ML + Perplexity + Burstiness combined

Author: Faisal
Date: November 6, 2024
"""

import os
import time
from combined_detector_hybrid import HybridDetector


class HybridTestSuite:
    """Test suite for hybrid detector"""

    def __init__(self):
        print("=" * 70)
        print("🧪 HYBRID DETECTOR TEST SUITE")
        print("=" * 70)
        print("\nInitializing detector...")
        self.detector = HybridDetector()

        self.results = {
            'ai_samples': {'correct': 0, 'incorrect': 0, 'total': 0},
            'human_samples': {'correct': 0, 'incorrect': 0, 'total': 0},
            'details': []
        }

    def load_samples(self, folder_path):
        """Load samples"""
        samples = []
        if not os.path.exists(folder_path):
            return samples

        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith('.txt'):
                filepath = os.path.join(folder_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        samples.append((filename, f.read()))
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

        return samples

    def test_ai_samples(self):
        """Test AI samples"""
        print("\n" + "=" * 70)
        print("📝 TESTING AI-GENERATED SAMPLES")
        print("=" * 70)

        samples = self.load_samples('tests/ai_samples')
        print(f"\nFound {len(samples)} AI samples\n")

        for filename, text in samples:
            result = self.detector.analyze(text)
            is_correct = result['final_decision']['is_ai_generated'] == True

            self.results['ai_samples']['total'] += 1
            if is_correct:
                self.results['ai_samples']['correct'] += 1
                status = "✅"
            else:
                self.results['ai_samples']['incorrect'] += 1
                status = "❌"

            self.results['details'].append({
                'filename': filename,
                'expected': 'AI',
                'predicted': result['final_decision']['label'],
                'confidence': result['final_decision']['confidence'],
                'correct': is_correct,
                'agreement': result['agreement']['agreement_level']
            })

            print(f"{status} {filename}: {result['final_decision']['label']} "
                  f"({result['final_decision']['confidence']:.1f}%) "
                  f"[{result['agreement']['agreement_level']}]")

    def test_human_samples(self):
        """Test human samples"""
        print("\n" + "=" * 70)
        print("📝 TESTING HUMAN-WRITTEN SAMPLES")
        print("=" * 70)

        samples = self.load_samples('tests/human_samples')
        print(f"\nFound {len(samples)} human samples\n")

        for filename, text in samples:
            result = self.detector.analyze(text)
            is_correct = result['final_decision']['is_ai_generated'] == False

            self.results['human_samples']['total'] += 1
            if is_correct:
                self.results['human_samples']['correct'] += 1
                status = "✅"
            else:
                self.results['human_samples']['incorrect'] += 1
                status = "❌"

            self.results['details'].append({
                'filename': filename,
                'expected': 'Human',
                'predicted': result['final_decision']['label'],
                'confidence': result['final_decision']['confidence'],
                'correct': is_correct,
                'agreement': result['agreement']['agreement_level']
            })

            print(f"{status} {filename}: {result['final_decision']['label']} "
                  f"({result['final_decision']['confidence']:.1f}%) "
                  f"[{result['agreement']['agreement_level']}]")

    def generate_report(self):
        """Generate final report"""
        print("\n" + "=" * 70)
        print("📊 FINAL TEST RESULTS - HYBRID SYSTEM")
        print("=" * 70)

        total = self.results['ai_samples']['total'] + self.results['human_samples']['total']
        correct = self.results['ai_samples']['correct'] + self.results['human_samples']['correct']

        overall_acc = (correct / total * 100) if total > 0 else 0
        ai_acc = (self.results['ai_samples']['correct'] / self.results['ai_samples']['total'] * 100) \
            if self.results['ai_samples']['total'] > 0 else 0
        human_acc = (self.results['human_samples']['correct'] / self.results['human_samples']['total'] * 100) \
            if self.results['human_samples']['total'] > 0 else 0

        print(f"\n📈 Overall Statistics:")
        print(f"   Total: {total}")
        print(f"   Correct: {correct}")
        print(f"   Accuracy: {overall_acc:.1f}%")

        print(f"\n🤖 AI Samples:")
        print(f"   Total: {self.results['ai_samples']['total']}")
        print(f"   Correct: {self.results['ai_samples']['correct']}")
        print(f"   Accuracy: {ai_acc:.1f}%")

        print(f"\n👤 Human Samples:")
        print(f"   Total: {self.results['human_samples']['total']}")
        print(f"   Correct: {self.results['human_samples']['correct']}")
        print(f"   Accuracy: {human_acc:.1f}%")

        # Misclassified samples
        incorrect_samples = [d for d in self.results['details'] if not d['correct']]
        if incorrect_samples:
            print(f"\n⚠️  Misclassified Samples ({len(incorrect_samples)}):")
            for sample in incorrect_samples[:10]:  # Show first 10
                print(f"   ❌ {sample['filename']}: Expected {sample['expected']}, "
                      f"Got {sample['predicted']} ({sample['confidence']:.1f}%) "
                      f"[{sample['agreement']}]")
        else:
            print("\n🎉 All samples classified correctly!")

        # Agreement analysis
        all_agree = [d for d in self.results['details'] if d.get('agreement') == 'All methods agree']
        print(f"\n🤝 Agreement Statistics:")
        print(f"   All Methods Agree: {len(all_agree)}/{total} ({len(all_agree)/total*100:.1f}%)")

        print("\n" + "=" * 70)

        # Save detailed report
        self._save_report(overall_acc, ai_acc, human_acc)

        return overall_acc

    def _save_report(self, overall_acc, ai_acc, human_acc):
        """Save report to file"""
        report_path = f'results/hybrid_test_report_{time.strftime("%Y%m%d_%H%M%S")}.txt'
        os.makedirs('results', exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("PROFESSIONAL HYBRID AI DETECTOR - TEST REPORT\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

            f.write("DETECTION METHODS:\n")
            f.write("1. Machine Learning (Custom trained RF classifier) - 50% weight\n")
            f.write("2. Perplexity Analysis (GPT-2 based) - 35% weight\n")
            f.write("3. Burstiness Analysis (Statistical) - 15% weight\n\n")

            f.write(f"OVERALL ACCURACY: {overall_acc:.2f}%\n")
            f.write(f"AI Samples Accuracy: {ai_acc:.2f}%\n")
            f.write(f"Human Samples Accuracy: {human_acc:.2f}%\n\n")

            f.write("=" * 70 + "\n")
            f.write("DETAILED RESULTS:\n")
            f.write("=" * 70 + "\n\n")

            for detail in self.results['details']:
                status = "✅ CORRECT" if detail['correct'] else "❌ INCORRECT"
                f.write(f"{status}: {detail['filename']}\n")
                f.write(f"   Expected: {detail['expected']}\n")
                f.write(f"   Predicted: {detail['predicted']} ({detail['confidence']:.2f}%)\n")
                f.write(f"   Agreement: {detail['agreement']}\n\n")

        print(f"📄 Detailed report saved: {report_path}")


if __name__ == "__main__":
    print("\n🚀 Starting Professional Hybrid Test Suite...\n")
    suite = HybridTestSuite()

    # Test AI samples
    suite.test_ai_samples()

    # Test human samples
    suite.test_human_samples()

    # Generate report
    accuracy = suite.generate_report()

    # Evaluation
    print("\n" + "=" * 70)
    print("🎯 ACCURACY EVALUATION")
    print("=" * 70)

    if accuracy >= 85:
        print("\n🎉 EXCELLENT! Production-ready accuracy!")
        print("✅ System is ready for deployment")
    elif accuracy >= 75:
        print("\n💪 GOOD! Above acceptable threshold!")
        print("✅ System is working well")
    elif accuracy >= 65:
        print("\n📈 ACCEPTABLE. Room for improvement.")
        print("⚠️ Consider collecting more training data")
    else:
        print("\n⚠️  NEEDS IMPROVEMENT")
        print("🔧 Train with more diverse samples")

    print("=" * 70 + "\n")
    print("✅ Testing complete!\n")
