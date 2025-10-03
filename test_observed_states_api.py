#!/usr/bin/env python3

import sys
import os
import json
sys.path.append('/home/philiposborne/Documents/elsciRL/elsciRL')

try:
    # Import only what we need for testing
    from elsciRL.application_suite.import_data import Applications
    from elsciRL.application_suite.import_tool import PullApplications
    
    print("Testing observed states API...")
    
    # Simulate WebApp initialization
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
    
    if not downloaded_apps:
        print("No applications downloaded. This is why observed states are not showing!")
        print("The WebApp needs to download applications first.")
        
        # Show what would be available if downloaded
        print("\nExpected observed states for each application:")
        for app_name, app_data in imports.items():
            if 'prerender_data_filenames' in app_data:
                prerender_files = app_data['prerender_data_filenames']
                print(f"  {app_name}: {list(prerender_files.keys())}")
    else:
        # Load application data
        application_data = PullApplications()
        pull_app_data = application_data.pull(problem_selection=downloaded_apps)
        
        print(f"Loaded data for applications: {list(pull_app_data.keys())}")
        
        # Test the get_observed_states function
        for app_name in downloaded_apps:
            if app_name in pull_app_data:
                app_data = pull_app_data[app_name]
                print(f"\n=== {app_name} ===")
                print(f"Data keys: {list(app_data.keys())}")
                
                if 'prerender_data' in app_data:
                    prerender_data = app_data['prerender_data']
                    observed_states = list(prerender_data.keys())
                    print(f"Observed states: {observed_states}")
                else:
                    print("No prerender_data found")
                    
        # Test the API response format
        print(f"\nAPI Response format for Classroom:")
        if 'Classroom' in downloaded_apps and 'Classroom' in pull_app_data:
            app_data = pull_app_data['Classroom']
            if 'prerender_data' in app_data:
                observed_states = list(app_data['prerender_data'].keys())
                api_response = {
                    'observedStates': {
                        'Classroom': observed_states
                    }
                }
                print(json.dumps(api_response, indent=2))
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
