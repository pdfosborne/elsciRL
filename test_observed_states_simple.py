#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/philiposborne/Documents/elsciRL/elsciRL')

try:
    from elsciRL.application_suite.import_data import Applications
    
    print("Testing observed states data structure...")
    
    # Get available applications
    imports = Applications().data
    possible_applications = list(imports.keys())
    print(f"Available applications: {possible_applications}")
    
    # Check prerender data for each application
    for app_name, app_data in imports.items():
        print(f"\n=== {app_name} ===")
        if 'prerender_data_filenames' in app_data:
            prerender_files = app_data['prerender_data_filenames']
            print(f"Prerender data filenames: {list(prerender_files.keys())}")
            for key, filename in prerender_files.items():
                print(f"  {key}: {filename}")
        else:
            print("No prerender_data_filenames found")
            
        if 'prerender_data_encoded_filenames' in app_data:
            encoded_files = app_data['prerender_data_encoded_filenames']
            print(f"Encoded data filenames: {list(encoded_files.keys())}")
        else:
            print("No prerender_data_encoded_filenames found")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
