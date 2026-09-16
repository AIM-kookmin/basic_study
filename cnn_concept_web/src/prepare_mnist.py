"""Generate lecture figures from actual MNIST and measured CNN outputs."""
from pathlib import Path
from mnist_lab import *

if __name__=='__main__':
    data=prepare_data()
    model,history,initial=fit_digit(data)
    result=evaluate_digit(model,data['x_test'],data['y_test'])
    assert result['accuracy']>.90
    assert int(result['confusion'].sum())==1000
    metrics=export_assets(Path(__file__).resolve().parents[1]/'assets',data,model,history,result)
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('test_indices','confusion','history')},ensure_ascii=False,indent=2))
