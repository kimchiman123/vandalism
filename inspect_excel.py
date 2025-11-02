import pandas as pd
import os

# 파일 경로 설정
file_path = r'C:\Users\sksg2\vandalism\open\1_생활인구분석_경기데이터드림.xlsx'
output_path = r'C:\Users\sksg2\vandalism\excel_analysis_output.txt'

# 키워드 설정
DAY_KEYWORD = '요일'
TIME_KEYWORD = '시간'

with open(output_path, 'w', encoding='utf-8') as f_out:
    if not os.path.exists(file_path):
        f_out.write(f"Error: File not found - {file_path}")
    else:
        try:
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
            f_out.write(f"Found sheets: {sheet_names}\n")

            sheet_day_name = None
            sheet_time_name = None

            for name in sheet_names:
                if DAY_KEYWORD in name:
                    sheet_day_name = name
                if TIME_KEYWORD in name:
                    sheet_time_name = name
            
            f_out.write(f"Dynamic sheet discovery: Day sheet='{sheet_day_name}', Time sheet='{sheet_time_name}'\n")
            f_out.write("="*50 + "\n")

            # --- 요일별 데이터 분석 ---
            if sheet_day_name:
                f_out.write(f"\n--- Analyzing Sheet: {sheet_day_name} ---\n")
                df_day = pd.read_excel(file_path, sheet_name=sheet_day_name)
                f_out.write("\n### Full Data ###\n")
                # .head()를 제거하여 전체 데이터 출력
                f_out.write(df_day.to_string())
                f_out.write("\n\n### Column List ###\n")
                f_out.write(str(df_day.columns.tolist()))
                f_out.write("\n" + "="*50 + "\n")
            else:
                f_out.write("\nDay analysis sheet not found.\n")

            # --- 시간대별 데이터 분석 ---
            if sheet_time_name:
                f_out.write(f"\n--- Analyzing Sheet: {sheet_time_name} ---\n")
                df_time = pd.read_excel(file_path, sheet_name=sheet_time_name)
                f_out.write("\n### Full Data ###\n")
                # .head()를 제거하여 전체 데이터 출력
                f_out.write(df_time.to_string())
                f_out.write("\n\n### Column List ###\n")
                f_out.write(str(df_time.columns.tolist()))
            else:
                f_out.write("\nTime analysis sheet not found.\n")

        except Exception as e:
            f_out.write(f"An error occurred while reading the file: {e}")

print(f"Analysis result saved to {output_path}")