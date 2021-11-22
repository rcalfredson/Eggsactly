import argparse
from glob import glob
import json
import os
import sys
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC

import timeit

sys.path.append(os.path.abspath("./"))
from lib.mail.send import send_mail


p = argparse.ArgumentParser(
    description="run automated browser test of the egg-counting tool"
)
p.add_argument("addr", help="url where the tool is deployed")
p.add_argument("test_type", choices=["watch", "regression"])
p.add_argument("upload_style", choices=["parallel", "serial"])
p.add_argument(
    "recipients",
    help="comma-separated list of email addresses"
    " where notifications should be sent if an error is detected",
)
opts = p.parse_args()
artifact_dest = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "test"
)
recipients = opts.recipients.split(",")
with open("configs/test_images.json", "r") as f:
    test_images = json.load(f)
imgs_to_test = {
    el: test_images[el]
    for el in test_images
    if opts.test_type in test_images[el]["test"]
}


class EggCountingTester:
    def __init__(self, opts):
        self.expected_title = "Egg Counting"
        profile = webdriver.FirefoxProfile()
        profile.set_preference("browser.download.folderList", 2)
        profile.set_preference("browser.download.manager.showWhenStarting", False)
        profile.set_preference("browser.download.dir", artifact_dest)
        profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/csv")
        firefox_opts = webdriver.FirefoxOptions()
        firefox_opts.add_argument("--headless")
        self.driver = webdriver.Firefox(profile, options=firefox_opts)
        self.opts = opts
        self.time_per_image_secs = 5.3

    def log(self, message):
        if type(message) is not str:
            message = str(message)
        self.messages.append(message)
        print(message)

    def wait(self, delay=10):
        return WebDriverWait(self.driver, delay)

    def concat_csvs(self):
        output_name = "combined_counts.csv"

        combined_csv = []
        for i, img_path in enumerate(self.img_list):
            with open(self.img_list[img_path]["csv"], "r") as f:
                file_contents = f.readlines()
                if i == 0:
                    combined_csv.extend(file_contents)
                else:
                    combined_csv.extend(file_contents[1:])
        output_abs_path = os.path.join(artifact_dest, output_name)
        with open(output_abs_path, "wt") as f:
            f.write("".join(combined_csv))
        self.csv_ref = output_abs_path

    def upload_images(self):
        self.num_uploaded = len(self.driver.find_elements_by_class_name("close-button"))
        self.driver.find_element_by_id("img-upload-btn").click()
        all_images_uploaded_message = self.wait(
            self.num_uploaded * self.time_per_image_secs
        ).until(
            EC.text_to_be_present_in_element(
                (By.ID, "updates-by-image"),
                f"Processing image {self.num_uploaded} of {self.num_uploaded}",
            )
        )
        assert (
            all_images_uploaded_message is not None
        ), "Did not find message reporting successful upload of all images"
        finished_counting_message = self.wait().until(
            EC.text_to_be_present_in_element(
                (By.ID, "updates-by-image"), "Finished counting eggs"
            )
        )
        assert (
            finished_counting_message is not None
        ), "Did not find message reporting successful egg counting"

    def select_image(self, img_path):
        upload_form = self.driver.find_element_by_name("img-upload-1")
        assert (
            upload_form is not None
        ), "Could not find <form> element for uploading images"
        upload_form.send_keys(img_path)

    def check_page_title(self):
        assert (
            self.expected_title in self.driver.title
        ), f'Could not find expected string "{expected_title}" in page title'

    def download_csv(self):
        download_csv_btn = self.wait().until(
            EC.element_to_be_clickable((By.ID, "download-csv"))
        )
        assert download_csv_btn is not None, "Could not locate CSV download button"
        download_csv_btn.click()

    def set_chamber_type(self, img_path):
        self.chamber_type = self.img_list[img_path]["type"]

    def check_images_v2(self, img_list):
        try:
            self.do_before_iterator(img_list)
            self.do_during_iterator()
            self.do_after_iterator()
        except Exception as exc:
            exc = str(exc)
            self.log(exc)
            self.errors.append(str(exc))
        finally:
            self.clean_up_files(artifact_dest)
        self.driver.close()
        self.send_notification()

    def do_before_iterator(self, img_list):
        self.img_list = img_list
        self.num_imgs = len(img_list)
        self.messages = []
        self.errors = []
        if self.opts.upload_style == "parallel":
            self.driver.get(opts.addr)
            self.check_page_title()
            self.concat_csvs()

    def do_after_iterator(self):
        if self.opts.upload_style == "parallel":
            self.upload_images()
            self.download_csv()
            self.check_csv()

    def do_during_iterator(self):
        for img_id in self.img_list:
            try:
                if self.opts.upload_style == "serial":
                    self.csv_ref = self.img_list[img_id]["csv"]
                    self.set_chamber_type(img_id)
                    self.driver.get(self.opts.addr)
                    self.check_page_title()
                self.select_image(self.img_list[img_id]["path"])
                if self.opts.upload_style == "serial":
                    self.upload_images()
                    self.download_csv()
                    self.check_csv(img_id)
            except Exception as exc:
                exc = str(exc)
                if img_id not in exc:
                    exc = self.add_prefix(exc, img_id)
                self.log(exc)
                self.errors.append(exc)
            finally:
                if self.opts.upload_style == "serial":
                    self.clean_up_files(artifact_dest)

    def get_latest_file(self):
        test_files = glob(os.path.join(artifact_dest, "*.csv"))
        abs_csv_ref = os.path.abspath(self.csv_ref)
        if abs_csv_ref in test_files:
            test_files.remove(abs_csv_ref)
        return max(test_files, key=os.path.getctime)

    @staticmethod
    def clean_up_files(dir_name):
        files = glob(os.path.join(dir_name, "*"))
        for f in files:
            os.remove(f)

    def send_notification(self):
        num_fails = len(self.errors)
        if num_fails == 0:
            print("All tests passed.")
            return
        message = (
            "An error" if num_fails == 1 else "Errors"
        ) + " occurred while testing the egg counting tool.\n\n"
        message += "\n".join(self.messages)
        send_mail(
            recipients,
            "Egg counting test error",
            message,
        )

    @staticmethod
    def is_image_filename(text_list, i):
        return "Egg Counter" in text_list[i - 1] or text_list[i - 1] == "\n"

    @staticmethod
    def num_imgs_in_csv(text_list):
        return len(
            [
                el
                for i, el in enumerate(text_list)
                if i > 0 and EggCountingTester.is_image_filename(text_list, i)
            ]
        )

    def log_csv_success(self, img_id):
        self.log(self.add_prefix("Egg counts match reference", img_id))

    def add_prefix(self, message, img_id):
        prefix = (
            ""
            if img_id is None
            else f"Image ID: {img_id}\tChamber type: {self.img_list[img_id]['type']}\n\t"
        )
        return f"{prefix}{message}"

    def compare_csv_for_single_img(self, cts1, cts2, img_id=None):
        prefix = (
            ""
            if img_id is None
            else f"Image ID: {img_id}\tChamber type: {self.img_list[img_id]['type']}\n\t"
        )
        assert len(set(cts1).difference(set(cts2))) == 0, self.add_prefix(
            "Diff found between generated egg counts and the given reference.", img_id
        )
        self.log_csv_success(img_id)

    def parse_csvs_by_image(self):
        csv_keys = ("generated", "reference")
        self.parsed_csvs = {k: {} for k in csv_keys}
        for i, text_list in enumerate([getattr(self, f"{k}_cts") for k in csv_keys]):
            k = csv_keys[i]
            current_img = None
            for j, line in enumerate(text_list):
                if j == 0:
                    continue
                if self.is_image_filename(text_list, j):
                    current_img = os.path.basename(line.strip())
                    self.parsed_csvs[k][current_img] = []
                else:
                    self.parsed_csvs[k][current_img].append(line)

    def compare_csv_for_multiple_imgs(self):
        self.parse_csvs_by_image()
        for i in range(self.num_generated_cts):
            try:
                img_id = list(self.img_list.keys())[i]
                filename = os.path.basename(self.img_list[img_id]["path"])
                self.set_chamber_type(img_id)

                self.compare_csv_for_single_img(
                    self.parsed_csvs["generated"][filename],
                    self.parsed_csvs["reference"][filename],
                    img_id,
                )
            except Exception as exc:
                exc = str(exc)
                self.log(exc)
                self.errors.append(exc)

    def check_csv(self, img_id=None):
        file_to_check = self.get_latest_file()
        with open(file_to_check, "r") as generated_cts, open(
            self.csv_ref, "r"
        ) as reference_cts:
            self.generated_cts, self.reference_cts = (
                generated_cts.readlines(),
                reference_cts.readlines(),
            )
            self.num_generated_cts, self.num_reference_cts = [
                self.num_imgs_in_csv(t)
                for t in (self.generated_cts, self.reference_cts)
            ]
            assert self.num_generated_cts == self.num_reference_cts, (
                "The generated egg counts contain different number of"
                " images from the reference"
            )
            if self.num_generated_cts == 1:
                self.compare_csv_for_single_img(
                    self.generated_cts, self.reference_cts, img_id=img_id
                )
            else:
                self.compare_csv_for_multiple_imgs()


if __name__ == "__main__":
    tester = EggCountingTester(opts)
    tester.check_images_v2(imgs_to_test)
