import json
import os
from datetime import datetime

class ResultsManager:
    def __init__(self, storage_file="analysis_results.json"):
        self.storage_file = storage_file
        self.load_results()
    
    def load_results(self):
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'r') as f:
                self.results = json.load(f)
        else:
            self.results = []
    
    def save_result(self, image_path, analysis_results):
        result_entry = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'results': analysis_results
        }
        
        self.results.append(result_entry)
        
        with open(self.storage_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def get_recent_results(self, count=5):
        return self.results[-count:] if self.results else []
    
    def get_results_by_grade(self, grade):
        return [r for r in self.results if r['results'].get('house_brackmann_grade') == grade]

# Usage example
if __name__ == "__main__":
    manager = ResultsManager()
    print(f"Stored results: {len(manager.results)}")