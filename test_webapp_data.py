#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/philiposborne/Documents/elsciRL/elsciRL')

try:
    from elsciRL.application_suite.import_data import Applications
    from elsciRL.application_suite.import_tool import PullApplications
    
    print("Testing WebApp data loading...")
    
    # Get available applications
    imports = Applications().data
    possible_applications = list(imports.keys())
    print(f"Available applications: {possible_applications}")
    
    # Check downloaded applications
    pull_apps = PullApplications()
    downloaded_apps = []
    for app_name in possible_applications:
        cache_dir = pull_apps._get_cache_dir(app_name)
        is_downloaded = os.path.exists(cache_dir) and os.path.exists(pull_apps._get_cache_metadata_file(app_name))
        if is_downloaded:
            downloaded_apps.append(app_name)
    
    print(f"Downloaded applications: {downloaded_apps}")
    
    if downloaded_apps:
        # Load application data
        application_data = PullApplications()
        pull_app_data = application_data.pull(problem_selection=downloaded_apps)
        
        print(f"Loaded data for applications: {list(pull_app_data.keys())}")
        
        for app_name in downloaded_apps:
            if app_name in pull_app_data:
                app_data = pull_app_data[app_name]
                print(f"\n=== {app_name} ===")
                print(f"Data keys: {list(app_data.keys())}")
                
                if 'prerender_data' in app_data:
                    prerender_data = app_data['prerender_data']
                    print(f"Prerender data keys: {list(prerender_data.keys())}")
                    for key, data in prerender_data.items():
                        print(f"  {key}: {type(data)} with {len(data) if hasattr(data, '__len__') else 'N/A'} items")
                else:
                    print("No prerender_data found")
    else:
        print("No applications downloaded - this is why observed states are not showing!")
        print("The WebApp needs to download applications first to have prerender data available.")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
